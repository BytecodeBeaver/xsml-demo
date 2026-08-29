import { Terminal } from '@xterm/xterm';
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
  description: string;
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
const gameScreen = getRequiredElement('game-screen');
const terminalElement = getRequiredElement('terminal');
const backButton = getRequiredElement('back-to-scenarios');

let terminal: Terminal | null = null;
let socket: Socket | null = null;
let inputBuffer = '';

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
      button.addEventListener('click', () => startGame(scenario.id));
      item.appendChild(button);
      scenarioList.appendChild(item);
    }
  } catch (error) {
    scenarioStatus.textContent = `Nie udalo sie pobrac listy scenariuszy: ${(error as Error).message}`;
  }
}

function startGame(scenarioId: string): void {
  scenarioScreen.hidden = true;
  gameScreen.hidden = false;

  if (!terminal) {
    terminal = new Terminal();
    terminal.open(terminalElement);
    terminal.onData(handleTerminalInput);
  } else {
    terminal.reset();
  }

  connectToGameServer(scenarioId);
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

function connectToGameServer(scenarioId: string): void {
  closeSocket();

  socket = io('/ws/game', { query: { scenario: scenarioId } });

  socket.on('connect', () => {
    terminal?.writeln(`Polaczono z serwerem gry (scenariusz: ${scenarioId})`);
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
  if (!terminal) {
    return;
  }

  terminal.writeln('');
  terminal.writeln('== Cele ==');
  for (const goal of state.goals) {
    writeGoal(goal, 0);
  }

  terminal.writeln('');
  terminal.writeln('== Dostepne akcje ==');
  for (const action of state.actions) {
    if (action.available) {
      terminal.writeln(`  ${action.id} - ${action.description ?? ''}`);
    }
  }
  terminal.write('\r\n> ');
}

function writeGoal(goal: Goal, depth: number): void {
  if (!terminal) {
    return;
  }
  const indent = '  '.repeat(depth);
  const mark = goal.achieved ? '[x]' : '[ ]';
  terminal.writeln(`${indent}${mark} ${goal.description || goal.id}`);
  for (const subgoal of goal.subgoals) {
    writeGoal(subgoal, depth + 1);
  }
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

void loadScenarios();
