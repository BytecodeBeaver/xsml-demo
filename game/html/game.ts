import { Terminal } from '@xterm/xterm';
import '@xterm/xterm/css/xterm.css';

const terminalElement = document.getElementById('terminal');

if (!terminalElement) {
  throw new Error('Terminal container was not found.');
}

const terminal = new Terminal();
terminal.open(terminalElement);
terminal.write('Hello from \x1B[1;3;31mxterm.js\x1B[0m $ ');