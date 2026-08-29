import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';
import { io, Socket } from 'socket.io-client';
import './style.css';

interface Scenario {
  id: string;
  title: string;
  description: string;
}

interface Goal {
  id: string;
  description: string | null;
  achieved: boolean;
  subgoals: Goal[];
}

interface ActionInfo {
  id: string;
  description: string;
  available: boolean;
}

interface GameState {
  goals: Goal[];
  actions: ActionInfo[];
}

function getRequiredElement(id: string): HTMLElement {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Required page element was not found: #${id}`);
  }
  return element;
}

const scenarioScreen = getRequiredElement('scenario-screen');
const scenarioList = getRequiredElement('scenario-list');
const scenarioStatus = getRequiredElement('scenario-status');
const nicknameInput = getRequiredElement('nickname-input') as HTMLInputElement;
const gameScreen = getRequiredElement('game-screen');
const activeNicknameLabel = getRequiredElement('active-nickname');
const terminalElement = getRequiredElement('terminal');
const backButton = getRequiredElement('back-to-scenarios');
const goalsList = getRequiredElement('goals-list');
const actionsList = getRequiredElement('actions-list');
const sideTabs = Array.from(document.querySelectorAll<HTMLButtonElement>('.side-tab'));
const sidePanels = Array.from(document.querySelectorAll<HTMLElement>('.side-panel-content'));

const NICKNAME_PATTERN = /^[A-Za-z0-9 _-]{1,24}$/;
const NICKNAME_STORAGE_KEY = 'game.nickname';
const TERMINAL_PROMPT = 'root@root:~# ';

let terminal: Terminal | null = null;
let fitAddon: FitAddon | null = null;
let resizeObserver: ResizeObserver | null = null;
let socket: Socket | null = null;
let inputBuffer = '';

function resizeTerminal(): void {
  if (!terminal || !fitAddon) {
    return;
  }

  const container = terminalElement.parentElement ?? terminalElement;
  const { width, height } = container.getBoundingClientRect();

  if (width === 0 || height === 0) {
    return;
  }

  fitAddon.fit();
}

function setupTerminal(): void {
  if (terminal) {
    resizeTerminal();
    return;
  }

  terminal = new Terminal({
    cursorBlink: true,
    scrollback: 2000,
    convertEol: true,
  });

  fitAddon = new FitAddon();
  terminal.loadAddon(fitAddon);
  terminal.open(terminalElement);
  terminal.onData(handleTerminalInput);

  resizeObserver = new ResizeObserver(() => {
    window.requestAnimationFrame(() => resizeTerminal());
  });

  resizeObserver.observe(terminalElement.parentElement ?? terminalElement);
  window.addEventListener('resize', resizeTerminal);
  requestAnimationFrame(() => resizeTerminal());
}

function restoreNickname(): void {
  const savedNickname = localStorage.getItem(NICKNAME_STORAGE_KEY);
  if (savedNickname) {
    nicknameInput.value = savedNickname;
  }
}

function setupTabs(): void {
  for (const tab of sideTabs) {
    tab.addEventListener('click', () => activateTab(tab.dataset.tab ?? ''));
  }
  activateTab('goals');
}

function activateTab(tabName: string): void {
  for (const tab of sideTabs) {
    tab.classList.toggle('active', tab.dataset.tab === tabName);
  }
  for (const panel of sidePanels) {
    panel.hidden = panel.dataset.panel !== tabName;
  }
}

async function loadScenarios(): Promise<void> {
  scenarioStatus.textContent = 'Ladowanie scenariuszy...';
  scenarioList.innerHTML = '';

  try {
    const response = await fetch('/api/scenarios');
    if (!response.ok) {
      throw new Error(`Serwer zwrocil status ${response.status}`);
    }
    const scenarios: Scenario[] = await response.json();

    if (scenarios.length === 0) {
      scenarioStatus.textContent = 'Brak dostepnych scenariuszy.';
      return;
    }

    scenarioStatus.textContent = '';
    for (const scenario of scenarios) {
      const item = document.createElement('li');
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'scenario-option';
      button.innerHTML = `<strong>${scenario.title}</strong><span>${scenario.description}</span>`;
      button.addEventListener('click', () => tryStartGame(scenario.id));
      item.appendChild(button);
      scenarioList.appendChild(item);
    }
  } catch (error) {
    scenarioStatus.textContent = `Nie udalo sie pobrac listy scenariuszy: ${(error as Error).message}`;
  }
}

function tryStartGame(scenarioId: string): void {
  const nickname = nicknameInput.value.trim();
  if (!NICKNAME_PATTERN.test(nickname)) {
    scenarioStatus.textContent = 'Podaj nick (1-24 znakow: litery, cyfry, spacja, - lub _).';
    nicknameInput.focus();
    return;
  }

  localStorage.setItem(NICKNAME_STORAGE_KEY, nickname);
  startGame(scenarioId, nickname);
}

function startGame(scenarioId: string, nickname: string): void {
  scenarioScreen.hidden = true;
  gameScreen.hidden = false;
  activeNicknameLabel.textContent = `Gracz: ${nickname}`;

  setupTerminal();
  terminal?.reset();

  connectToGameServer(scenarioId, nickname);
}

function handleTerminalInput(data: string): void {
  if (!terminal) {
    return;
  }

  if (data === '\r') {
    terminal.write('\r\n');
    sendAction(inputBuffer.trim());
    inputBuffer = '';
    return;
  }

  if (data === '\u007f') {
    if (inputBuffer.length > 0) {
      inputBuffer = inputBuffer.slice(0, -1);
      terminal.write('\b \b');
    }
    return;
  }

  inputBuffer += data;
  terminal.write(data);
}

function sendAction(actionId: string): void {
  if (!actionId || !socket || !socket.connected) {
    return;
  }
  socket.emit('action', { action_id: actionId });
}

function connectToGameServer(scenarioId: string, nickname: string): void {
  closeSocket();

  socket = io('/ws/game', { query: { scenario: scenarioId, nickname } });

  socket.on('connect', () => {
    terminal?.writeln(`Polaczono z serwerem gry (scenariusz: ${scenarioId})`);
  });

  socket.on('terminal_output', (payload: { command: string; simulated_output: string }) => {
    if (!terminal) {
      return;
    }

    terminal.write(`${payload.command}\r\n`);
    if (payload.simulated_output) {
      terminal.write(payload.simulated_output.replace(/\n/g, '\r\n'));
    }
  });

  socket.on('state', (state: GameState) => {
    renderState(state);
  });

  socket.on('error', (payload: { message: string }) => {
    terminal?.writeln(`\r\nBlad: ${payload.message}`);
  });

  socket.on('disconnect', () => {
    terminal?.writeln('\r\nPolaczenie z serwerem gry zostalo zamkniete.');
  });

  socket.on('connect_error', () => {
    terminal?.writeln('\r\nNie udalo sie polaczyc z serwerem gry.');
  });
}

function renderState(state: GameState): void {
  goalsList.innerHTML = '';
  for (const goal of state.goals) {
    goalsList.appendChild(buildGoalNode(goal));
  }

  actionsList.innerHTML = '';
  for (const action of state.actions.filter((candidate) => candidate.available)) {
    actionsList.appendChild(buildActionItem(action));
  }

  terminal?.write(`\r\n${TERMINAL_PROMPT}`);
}

// The full goal hierarchy is always visible; only a goal's own description stays hidden until achieved.
function buildGoalNode(goal: Goal): HTMLLIElement {
  const item = buildGoalItem(goal);
  if (goal.subgoals.length > 0) {
    const sublist = document.createElement('ul');
    sublist.className = 'goal-subgoals';
    for (const subgoal of goal.subgoals) {
      sublist.appendChild(buildGoalNode(subgoal));
    }
    item.appendChild(sublist);
  }
  return item;
}

function buildGoalItem(goal: Goal): HTMLLIElement {
  const item = document.createElement('li');
  const label = document.createElement('label');
  label.className = `goal-item${goal.achieved ? ' achieved' : ' locked'}`;

  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.checked = goal.achieved;
  checkbox.disabled = true;

  const text = document.createElement('span');
  // The server only sends a description once the goal's content may be revealed.
  text.textContent = goal.description ?? '??? (cel ukryty)';

  label.appendChild(checkbox);
  label.appendChild(text);
  item.appendChild(label);

  return item;
}

function buildActionItem(action: ActionInfo): HTMLLIElement {
  const item = document.createElement('li');
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'action-item';
  button.textContent = action.description || action.id;
  button.addEventListener('click', () => sendAction(action.id));
  item.appendChild(button);
  return item;
}

function closeSocket(): void {
  if (socket) {
    socket.removeAllListeners();
    socket.close();
    socket = null;
  }
}

function returnToScenarioSelect(): void {
  closeSocket();
  gameScreen.hidden = true;
  scenarioScreen.hidden = false;
  void loadScenarios();
}

backButton.addEventListener('click', returnToScenarioSelect);

restoreNickname();
setupTabs();
void loadScenarios();
