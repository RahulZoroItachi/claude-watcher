#!/usr/bin/env node

const { execSync, spawnSync } = require('child_process');
const path = require('path');

const PORT = process.env.PORT || 7755;
const APP  = path.join(__dirname, '..', 'app.py');

console.log('\n  claude-watcher');
console.log('  Analyze your Claude Code costs and usage patterns.');
console.log('  Your data never leaves this machine.\n');

// ── Find Python 3 ─────────────────────────────────────────────────────────────
let python = null;
for (const candidate of ['python3', 'python']) {
  try {
    const v = execSync(`${candidate} --version 2>&1`).toString();
    if (v.startsWith('Python 3')) { python = candidate; break; }
  } catch {}
}
if (!python) {
  console.error('  Error: Python 3 is required.\n  Install from https://python.org\n');
  process.exit(1);
}

// ── Ensure pip dependencies ───────────────────────────────────────────────────
for (const dep of ['streamlit', 'pandas']) {
  try {
    execSync(`${python} -c "import ${dep}"`, { stdio: 'ignore' });
  } catch {
    process.stdout.write(`  Installing ${dep}... `);
    try {
      execSync(`${python} -m pip install ${dep} -q`, { stdio: 'ignore' });
    } catch {
      execSync(`${python} -m pip install ${dep} -q --break-system-packages`, { stdio: 'ignore' });
    }
    console.log('done');
  }
}

// ── Launch ────────────────────────────────────────────────────────────────────
console.log(`  Open http://localhost:${PORT}\n`);

const result = spawnSync(python, [
  '-m', 'streamlit', 'run', APP,
  '--server.port',            String(PORT),
  '--browser.gatherUsageStats', 'false',
  '--server.enableCORS',      'false',
], { stdio: 'inherit' });

process.exit(result.status ?? 0);
