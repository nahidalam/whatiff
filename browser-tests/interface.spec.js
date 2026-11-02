import { test, expect } from '@playwright/test';
import fs from 'node:fs/promises';

test.beforeEach(async () => { await fs.mkdir('data/browser-evidence', { recursive: true }); });

test('saved replay, image comparison, evidence and JSON export', async ({ page }) => {
  const rehearsalStarted = Date.now();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto('/');
  await page.getByRole('button', { name: /Replay saved example/ }).click();
  await expect(page.getByText('Examples', { exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Decision recorded. Review pending.' })).toBeVisible();
  await expect(page.getByText('Recorded live Meta decisions · Replay: 0 model calls', { exact: true })).toBeVisible();
  const execution = page.getByRole('region', { name: 'Live execution', exact: true });
  await expect(execution).toBeVisible();
  await expect(execution.getByRole('article')).toHaveCount(3);
  await expect(execution.getByRole('article', { name: 'Policy execution', exact: true })).toHaveCount(0);
  const evaluator = execution.getByRole('article', { name: 'Evaluator execution', exact: true });
  await expect(evaluator).toContainText('Policy testing · Muse Spark 1.34 calls · Completed');
  await expect(evaluator).toContainText('Scoring · Deterministic verificationCompleted');

  await expect(execution.getByText('Saved replay · 8 recorded calls · 0 new calls')).toBeVisible();
  const generator = page.getByRole('article', { name: 'Generator execution' });
  await expect(generator).toContainText('Muse Image3 calls · Completed');
  await expect(generator).toContainText('SAM 3.1Skipped');

  const images = page.locator('.comparison img');
  await expect(images).toHaveCount(2);
  await expect.poll(() => images.evaluateAll(nodes => nodes.every(node => node.complete && node.naturalWidth > 0))).toBe(true);
  await expect(page.getByText('Task: “Pick up the sponge to clean the dish.”', { exact: true })).toBeVisible();
  await expect(page.getByRole('tab')).toHaveCount(3);
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('link', { name: 'Export JSON' }).click();
  const download = await downloadPromise;
  const report = JSON.parse(await fs.readFile(await download.path(), 'utf8'));
  expect(report.mode).toBe('prepared_example');
  expect(report.scenarios[0].evaluation_outcome).toBe('inconclusive');
  expect(report.task).toBe('Pick up the sponge to clean the dish.');
  await page.getByRole('button', { name: /Replay saved example/ }).click();
  await expect(page.getByText('Examples', { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
  const status = await (await page.request.get('/api/status')).json();
  if (status.meta_configured && status.baseten_configured) await expect(page.getByRole('button', { name: 'Run Experiments' })).toBeEnabled();
  else await expect(page.getByRole('button', { name: 'Run Experiments' })).toBeDisabled();
  const folderLink = page.locator('.saved-files a');
  await expect(folderLink).toContainText('/data/web/');
  await expect(folderLink).toHaveAttribute('href', /^file:\/\//);
  await page.route('**/api/runs/*/reveal', route => route.fulfill({ json: { path: 'mock-folder' } }));
  const revealed = page.waitForRequest(request => request.url().endsWith('/reveal') && request.method() === 'POST');
  await folderLink.click();
  await revealed;
  await page.screenshot({ path: 'data/browser-evidence/whatiff-desktop.png', fullPage: true });
  await fs.writeFile('data/browser-evidence/replay-rehearsal.json', JSON.stringify({ mode: 'recorded', model_calls: 0, walkthrough_seconds: (Date.now() - rehearsalStarted) / 1000, steps: ['Open recorded example', 'Inspect image pair and highlight', 'Read decisions and outcome', 'Inspect validity', 'Export JSON', 'Replay saved example'] }, null, 2));
});

test('mobile comparison stays within the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.getByRole('button', { name: /Replay saved example/ }).click();
  await expect(page.getByText('Examples', { exact: true })).toBeVisible();
  await expect(page.locator('.comparison img')).toHaveCount(2);
  const dimensions = await page.evaluate(() => ({ content: document.documentElement.scrollWidth, viewport: innerWidth }));
  expect(dimensions.content).toBeLessThanOrEqual(dimensions.viewport);
  await page.screenshot({ path: 'data/browser-evidence/whatiff-mobile.png', fullPage: true });
});

test('live Meta washing-dishes pipeline through the interface', async ({ page }) => {
  test.skip(process.env.WHATIFF_LIVE_TEST !== '1', 'Explicit opt-in: this test makes live Meta API calls.');
  const rehearsalStarted = Date.now();
  await page.goto('/');
  await page.getByRole('button', { name: /Replay saved example/ }).click();
  await expect(page.getByText('Examples', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Run Experiments' }).click();
  await expect(page.getByRole('status')).toBeVisible();
  await expect(page.getByText('LIVE RUN', { exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Compare the observation and decision.' })).toBeVisible({ timeout: 230000 });
  await expect(page.getByText('Saved demo · recorded Meta decisions · no API calls')).toHaveCount(0);
  const exportPromise = page.waitForEvent('download');
  await page.getByRole('link', { name: 'Export JSON' }).click();
  const download = await exportPromise;
  const report = JSON.parse(await fs.readFile(await download.path(), 'utf8'));
  expect(report.mode).toBe('live');
  expect(report.task).toBe('Pick up the sponge to clean the dish.');
  expect(report.scenarios).toHaveLength(3);
  expect(report.policy_model).toBe('meta-models/muse-spark-open-1026');
  expect(report.policy_provider).toBe('baseten');
  await expect(page.getByRole('region', { name: 'Live execution' })).toContainText('Live · 8 recorded calls');
  const runId = report.evidence_id.split(':')[1];
  const actualOriginal = await page.request.get(`/api/runs/${runId}/images/original.png`);
  expect(await actualOriginal.body()).toEqual(await fs.readFile('public/examples/washing-dishes.jpg'));
  const internal = await (await page.request.get(`/api/runs/${runId}`)).json();
  for (const item of internal.cases) expect(item.image_dimensions.original).toEqual([960, 540]);
  await expect(page.getByRole('article', { name: 'Evaluator execution' })).toContainText('Muse Spark Open · Baseten4 calls · Completed');
  await expect(page.getByRole('article', { name: 'Adversary execution' })).not.toContainText('Baseten');

  await fs.writeFile('data/browser-evidence/live-report.json', JSON.stringify(report, null, 2));
  await page.screenshot({ path: 'data/browser-evidence/whatiff-live.png', fullPage: true });
  await fs.writeFile('data/browser-evidence/live-rehearsal.json', JSON.stringify({ mode: 'live', walkthrough_seconds: (Date.now() - rehearsalStarted) / 1000, observed_actions: [report.original.observed_action, report.scenarios[0].observed_action], outcome: report.scenarios[0].evaluation_outcome, steps: ['Open recorded example', 'Start one live run', 'Read actual decisions', 'Inspect three contributions', 'Export actual JSON report'] }, null, 2));
});

test('potential failures and inconclusive cases stay distinct', async ({ page }) => {
  await page.route('**/api/replay', async route => {
    const response = await route.fetch();
    const data = await response.json();
    const sample = data.cases[0];
    data.cases = [
      sample,
      { ...sample, id: '1', name: 'Distractor', outcome: 'potential_failure', explanation: 'Mock test: action violates a supported criterion.' },
      { ...sample, id: '2', name: 'Relocation', outcome: 'inconclusive', explanation: 'Mock test: intervention is unverified.', validity: { ...sample.validity, status: 'unverified' } },
    ];
    await route.fulfill({ response, json: data });
  });
  await page.goto('/');
  await page.getByRole('button', { name: /Replay saved example/ }).click();
  await expect(page.getByRole('tab')).toHaveCount(3);
  await page.getByRole('tab', { name: /Distractor/ }).click();
  await expect(page.getByRole('heading', { name: 'A potential action-selection failure.' })).toBeVisible();
  await page.getByRole('tab', { name: /Relocation/ }).click();
  await expect(page.getByRole('heading', { name: 'Decision recorded. Review pending.' })).toBeVisible();
});


test('real egocentric input is the default, without model calls', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Generate Scenarios. Test Robot Behavior. Prevent Real-World Failures.', exact: true })).toBeVisible();
  await expect(page.getByText('Examples', { exact: true })).toBeVisible();
  const input = page.locator('.comparison img');
  await expect(input).toHaveCount(1);
  await expect.poll(() => input.evaluate(node => node.complete && node.naturalWidth === 960)).toBe(true);
  await page.getByText('Change task, image or generator').click();
  await expect(page.locator('#task')).toHaveValue('Pick up the sponge to clean the dish.');
  await expect(page.locator('#backend')).toHaveValue('muse');
  await expect(page.locator('#policy-provider')).toHaveValue('baseten');
  await expect(page.getByRole('button', { name: 'Washing dishes', exact: true })).toBeVisible();
  await expect(page.getByText('Saved demo · recorded Meta decisions · no API calls')).toHaveCount(0);
  await page.getByText('Change task, image or generator').click();
  await page.getByRole('button', { name: 'Cutting vegetables', exact: true }).click();
  await page.getByText('Change task, image or generator').click();
  await expect(page.locator('#task')).toHaveValue('Pick up the knife to cut the carrots.');
  await page.getByText('Change task, image or generator').click();
  await page.getByRole('button', { name: 'Washing dishes', exact: true }).click();
  await expect(page.locator('.edit-step')).toHaveCount(0);
  await page.screenshot({ path: 'data/browser-evidence/whatiff-real-input.png', fullPage: true });
});


test('human review stays available and preserves checks in export', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: /Replay saved example/ }).click();
  await expect(page.getByRole('region', { name: 'Review result' })).toBeVisible();
  await page.getByRole('textbox', { name: 'Reviewer', exact: true }).fill('Demo reviewer');
  await expect(page.getByRole('button', { name: 'Approve edit', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Approve edit', exact: true }).click();
  await expect(page.getByText('Edit approved', { exact: true })).toBeVisible();
  await expect(page.getByText('Expected actions pending', { exact: true })).toBeVisible();
  await expect(page.getByText('Checks failed', { exact: true })).toBeVisible();
  await page.getByRole('textbox', { name: 'Edit review note' }).fill('Reject: image dimensions changed.');
  await page.getByRole('button', { name: 'Reject edit', exact: true }).click();
  await expect(page.getByText('Edit rejected', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Save expected actions & evaluate' })).toBeDisabled();
  const panel = page.getByRole('region', { name: 'Review result' });
  await panel.getByRole('group', { name: 'Original image', exact: true }).getByLabel('Pick target', { exact: true }).check();
  await panel.getByRole('group', { name: 'Edited image', exact: true }).getByLabel('Pick target', { exact: true }).check();
  await page.getByRole('textbox', { name: 'Expected action rationale' }).fill('The sponge is identifiable in both images; target selection is appropriate.');
  await page.getByRole('button', { name: 'Save expected actions & evaluate' }).click();
  await expect(page.getByText('Expected actions approved and saved.')).toBeVisible();
  const pending = page.waitForEvent('download');
  await page.getByRole('link', { name: 'Export JSON' }).click();
  const download = await pending;
  const report = JSON.parse(await fs.readFile(await download.path(), 'utf8'));
  expect(report.scenarios[0].edit_approval).toBe('rejected');
  expect(report.scenarios[0].criterion_approval).toBe('approved');
  expect(report.scenarios[0].verification_status).toBe('invalid');
  expect(report.review).toBeUndefined();
  expect(report.scenarios[0].image).toBe('counterfactual-0.png');
  expect(report.scenarios[0].evaluation_outcome).toBe('inconclusive');
  await page.getByRole('tab', { name: 'Distractor', exact: true }).click();
  await page.getByRole('tab', { name: 'Occlusion', exact: true }).click();
  await expect(page.getByText('Edit rejected', { exact: true })).toBeVisible();
});


test('running live after replay sends the displayed original image', async ({ page }) => {
  let uploaded;
  await page.route('**/api/status', route => route.fulfill({ json: { meta_configured: true, baseten_configured: true, busy: false } }));
  await page.route('**/api/runs', async route => {
    uploaded = route.request().postDataBuffer();
    await route.fulfill({ status: 400, json: { detail: 'Mock upload captured; no model calls.' } });
  });
  await page.goto('/');
  await page.getByRole('button', { name: /Replay saved example/ }).click();
  await expect(page.locator('.comparison img')).toHaveCount(2);
  await page.getByRole('button', { name: 'Run Experiments' }).click();
  await expect(page.getByRole('alert')).toContainText('Mock upload captured');
  expect(uploaded.includes(await fs.readFile('public/examples/washing-dishes.jpg'))).toBe(true);
});


test('desktop headline is one line and run button is below settings', async ({ page }) => {
  await page.setViewportSize({ width: 1069, height: 900 });
  await page.goto('/');
  const heading = page.getByRole('heading', { level: 1 });
  const layout = await heading.evaluate(node => { const range = document.createRange(); range.selectNodeContents(node); return { lines: range.getClientRects().length, width: document.documentElement.scrollWidth, viewport: innerWidth }; });
  expect(layout.lines).toBe(1);
  expect(layout.width).toBeLessThanOrEqual(layout.viewport);
  const settings = await page.locator('.demo-settings').boundingBox();
  const button = await page.getByRole('button', { name: 'Run Experiments' }).boundingBox();
  expect(button.y).toBeGreaterThanOrEqual(settings.y + settings.height);
  await page.screenshot({ path: 'data/browser-evidence/whatiff-headline-layout.png', fullPage: true });
});


test('live policy calls are Evaluator substeps with consistent statuses', async ({ page }) => {
  let recorded;
  let policyState = 'waiting';
  await page.route('**/api/replay', async route => {
    const response = await route.fetch();
    recorded = await response.json();
    recorded.execution.mode = 'live';
    recorded.execution.model_calls_this_execution = 2;
    recorded.execution.recorded_model_calls = 0;
    const policy = recorded.execution.stages.find(stage => stage.role === 'Policy');
    policy.status = policyState;
    policy.tools[0].calls = 1;
    const evaluator = recorded.execution.stages.find(stage => stage.role === 'Evaluator');
    evaluator.status = 'waiting';
    evaluator.tools[0].status = 'waiting';
    await route.fulfill({ response, json: recorded });
  });
  await page.goto('/');
  await page.getByRole('button', { name: /Replay saved example/ }).click();
  const panel = page.getByRole('region', { name: 'Live execution', exact: true });
  await expect(panel.getByRole('article')).toHaveCount(3);
  await expect(panel).toContainText('Live · 2 recorded calls');
  const evaluator = panel.getByRole('article', { name: 'Evaluator execution' });
  await expect(evaluator.locator('.execution-status')).toHaveText('In progress');
  await expect(evaluator).toContainText('Policy testing · Muse Spark 1.31 call · Completed');
  await expect(evaluator).toContainText('Scoring · Deterministic verificationWaiting');
  for (const [state, label] of [['running', 'Running'], ['failed', 'Failed']]) {
    policyState = state;
    await page.getByRole('button', { name: /Replay saved example/ }).click();
    await expect(evaluator.locator('.execution-status')).toHaveText(label);
    await expect(evaluator).toContainText(`Policy testing · Muse Spark 1.31 call · ${label}`);
  }
});

test('policy selection is explicit and Baseten failure is visible without fallback', async ({ page }) => {
  let provider;
  await page.route('**/api/status', route => route.fulfill({ json: { meta_configured: true, baseten_configured: true, busy: false } }));
  await page.route('**/api/runs', route => {
    const body = route.request().postDataBuffer().toString();
    provider = body.includes('\r\n\r\nbaseten\r\n') ? 'baseten' : 'meta';
    return route.fulfill({ status: 400, json: { detail: 'Baseten: HTTP 503' } });
  });
  await page.goto('/');
  await page.locator('.demo-settings summary').click();
  await expect(page.locator('#policy-provider')).toHaveValue('baseten');
  await page.getByRole('button', { name: 'Run Experiments' }).click();
  await expect(page.getByRole('alert')).toContainText('Baseten: HTTP 503');
  expect(provider).toBe('baseten');
  await expect(page.locator('#policy-provider')).toHaveValue('baseten');
  await page.locator('#policy-provider').selectOption('meta');
  await expect(page.locator('#policy-provider')).toHaveValue('meta');
});

test('Baseten Evaluator shows one policy tool and separate scoring', async ({ page }) => {
  await page.route('**/api/replay', async route => {
    const response = await route.fetch();
    const data = await response.json();
    data.execution.mode = 'live';
    data.execution.model_calls_this_execution = 4;
    const policy = data.execution.stages.find(stage => stage.role === 'Policy');
    policy.tools = [{ name: 'Muse Spark Open · Baseten', model: 'meta-models/muse-spark-open-1026', calls: 4, status: 'completed' }];
    await route.fulfill({ response, json: data });
  });
  await page.goto('/');
  await page.getByRole('button', { name: /Replay saved example/ }).click();
  const evaluator = page.getByRole('article', { name: 'Evaluator execution' });
  await expect(evaluator.locator('small')).toHaveCount(2);
  await expect(evaluator).toContainText('Policy testing · Muse Spark Open · Baseten4 calls · Completed');
  await expect(evaluator).not.toContainText('Muse Spark 1.3');
  await expect(evaluator).toContainText('Scoring · Deterministic verificationCompleted');
});
