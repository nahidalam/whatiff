import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const labels = { pass: 'Pass', potential_failure: 'Potential Failure', inconclusive: 'Needs review' };
const samples = [
  { name: 'Washing dishes', image: 'washing-dishes.jpg', task: 'Pick up the sponge to clean the dish.' },
  { name: 'Cutting vegetables', image: 'cutting-vegetables.jpg', task: 'Pick up the knife to cut the carrots.' },
  { name: 'Cooking', image: 'cooking.jpg', task: 'Pick up the spoon to stir the food.' },
];
const action = value => value ? value.replaceAll('_', ' ') : 'Awaiting decision';

function Icon({ name, size = 18 }) {
  const paths = { play: <path d="m9 5 10 7-10 7V5Z"/>, arrow: <path d="M4 12h16m-6-6 6 6-6 6"/>, download: <><path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/></>, upload: <><path d="M12 17V3m-5 5 5-5 5 5M4 16v5h16v-5"/></>, check: <path d="m5 12 4 4L19 6"/>, repeat: <><path d="M20 7h-8m8 0V2M4 17h8m-8 0v5"/><path d="M20 7a8 8 0 0 0-14-2M4 17a8 8 0 0 0 14 2"/></>, eye: <><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/></> };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The request could not be completed.');
  return data;
}

function Outcome({ value }) {
  return <span className={`outcome ${value || 'pending'}`}>{value === 'pass' && <Icon name="check" size={13}/>} {labels[value] || 'Pending'}</span>;
}

function Observation({ kind, image, data, mode, loading, region, task }) {
  const [dimensions, setDimensions] = useState(null);
  return <section className={`observation ${kind === 'Counterfactual' ? 'edited' : ''}`} aria-label={kind}>
    <div className="image-heading"><div><span className="index">{kind === 'Original' ? '00' : '01'}</span><strong>{kind === 'Original' ? '1. Input image' : '2. Counterfactual image'}</strong></div></div>
    <div className="image-frame">
      {image ? <img src={image} onLoad={event => setDimensions({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} alt={kind === 'Original' ? 'Original input observation' : 'Counterfactual observation with the selected intervention'}/> : <div className="image-placeholder"><Icon name="eye" size={30}/><span>{loading ? 'Building the counterfactual…' : 'Click Run Experiments to create the edited image'}</span></div>}
      {image && region && dimensions && <svg className="region-overlay" viewBox={`0 0 ${dimensions.width} ${dimensions.height}`} preserveAspectRatio="xMidYMid meet" aria-hidden="true"><rect className="region-halo" x={region[0] - 6} y={region[1] - 6} width={region[2] - region[0] + 12} height={region[3] - region[1] + 12} rx="5"/><rect className={`region-border ${kind === 'Original' ? 'reference-border' : ''}`} x={region[0] - 6} y={region[1] - 6} width={region[2] - region[0] + 12} height={region[3] - region[1] + 12} rx="5"/><path className="region-leader" d={`M${region[0] - 6},${region[1] - 6} l-36,-36 h-100`}/><text className="region-label" x={region[0] - 140} y={region[1] - 56}>{kind === 'Original' ? 'REFERENCE REGION' : 'PARTIAL OCCLUSION'}</text></svg>}
      {image && <span className="frame-tag">{kind === 'Original' ? 'INPUT' : 'COUNTERFACTUAL'}</span>}
    </div>
    {kind === 'Original' ? <div className="decision-row"><div><span className="small-label">TASK</span><strong className="decision">{task}</strong></div></div> : <div className="decision-row"><div><span className="small-label">{data?.policy_model === 'meta-models/muse-spark-open-1026' ? 'MUSE SPARK OPEN · BASETEN' : 'MUSE SPARK'} · {mode === 'prepared_example' ? 'RECORDED DECISION' : 'OBSERVED DECISION'}</span><strong className="decision">{action(data?.observed_action)}</strong></div><Outcome value={data?.outcome}/></div>}
    {data && kind !== 'Original' && <><details className="policy-reason"><summary>Why this action?</summary><p>{data.policy_reason}</p></details></>}
  </section>;
}

function ExecutionPanel({ execution }) {
  if (!execution) return null;
  const names = { completed: 'Completed', running: 'Running', in_progress: 'In progress', waiting: 'Waiting', skipped: 'Skipped', failed: 'Failed' };
  const records = Object.fromEntries(execution.stages.map(stage => [stage.role, stage]));
  function aggregate(statuses) {
    if (statuses.includes('failed')) return 'failed';
    if (statuses.includes('running')) return 'running';
    if (statuses.length && statuses.every(status => ['completed', 'skipped'].includes(status))) return statuses.includes('completed') ? 'completed' : 'skipped';
    return statuses.includes('completed') || statuses.includes('in_progress') ? 'in_progress' : 'waiting';
  }
  function agent(role, summary) {
    const record = records[role];
    const tools = record?.tools || [];
    const status = record?.status === 'running' || record?.status === 'failed' ? record.status : aggregate([record?.status || 'waiting', ...tools.map(tool => tool.status)]);
    return { role, summary, status, tools };
  }
  const policy = records.Policy?.tools || [];
  const scoring = records.Evaluator?.tools || [];
  const evaluatorTools = [...policy.map(tool => ({ ...tool, status: ['running', 'failed'].includes(records.Policy?.status) ? records.Policy.status : tool.status, label: `Policy testing · ${tool.name}` })), ...scoring.map(tool => ({ ...tool, status: ['running', 'failed'].includes(records.Evaluator?.status) ? records.Evaluator.status : tool.status, label: `Scoring · ${tool.name}` }))];
  const stages = [agent('Adversary', 'Scenario planning'), agent('Generator', 'Scene editing and optional grounding'),
    { role: 'Evaluator', summary: 'Independent policy testing and scoring', tools: evaluatorTools, status: aggregate(evaluatorTools.map(tool => tool.status)) }];
  return <section className="execution-panel" aria-label="Live execution"><header><strong>Live execution</strong><span>{execution.mode === 'saved_replay' ? `Saved replay · ${execution.recorded_model_calls} recorded calls · 0 new calls` : `Live · ${execution.model_calls_this_execution} recorded calls`}</span></header><div className="execution-stages">{stages.map(stage => <article key={stage.role} aria-label={`${stage.role} execution`}><div><strong>{stage.role}</strong><span className={`execution-status ${stage.status}`}>{names[stage.status]}</span></div><p>{stage.summary}</p>{stage.tools.map(tool => <small key={tool.label || tool.name}>{tool.label || tool.name}<span>{tool.calls ? `${tool.calls} call${tool.calls === 1 ? '' : 's'} · ` : ''}{names[tool.status]}</span></small>)}</article>)}</div></section>;
}

function ReviewPanel({ run, item, onSaved }) {
  const [reviewer, setReviewer] = useState('');
  const [note, setNote] = useState('');
  const [reason, setReason] = useState('');
  const [before, setBefore] = useState([]);
  const [after, setAfter] = useState([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [failed, setFailed] = useState(false);
  async function save(values) {
    setBusy(true); setMessage(''); setFailed(false);
    try {
      const updated = await api(`/api/runs/${run.id}/cases/${item.id}/review`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reviewer, ...values }) });
      onSaved(updated); setMessage('Review saved. JSON export updated.');
    } catch (error) { setMessage(error.message); setFailed(true); }
    finally { setBusy(false); }
  }
  const choices = (values, setter) => ['PICK_TARGET', 'REPOSITION', 'ABSTAIN'].map(value => <label key={value}><input type="checkbox" checked={values.includes(value)} onChange={event => setter(event.target.checked ? [...values, value] : values.filter(item => item !== value))}/>{ { PICK_TARGET: 'Pick target', REPOSITION: 'Change viewpoint / approach', ABSTAIN: 'Decline pickup' }[value]}</label>);
  return <section className="review-panel" aria-label="Review result">
    <div className="review-title"><h3>Review result</h3><div className="approval-states"><span>{item.review?.edit_status ? `Edit ${item.review.edit_status}` : 'Edit not reviewed'}</span><span>{item.criterion_approved ? 'Expected actions approved' : 'Expected actions pending'}</span><span>{item.validity.status === 'invalid' ? 'Checks failed' : item.validity.status === 'valid' ? 'Edit checks passed' : 'Edit verification pending'}</span></div></div>

    <label className="reviewer-label">Reviewer<input aria-label="Reviewer" value={reviewer} maxLength={200} placeholder="Your name" onChange={event => setReviewer(event.target.value)} disabled={busy}/></label>
    <div className="review-grid"><form onSubmit={event => event.preventDefault()}><h4>1. Check the image edit</h4><textarea aria-label="Edit review note" placeholder="Optional review note" value={note} maxLength={2000} onChange={event => setNote(event.target.value)}/><div className="review-buttons"><button type="button" disabled={busy || !reviewer.trim()} onClick={() => save({ edit_status: 'approved', edit_note: note })}>Approve edit</button><button type="button" disabled={busy || !reviewer.trim()} onClick={() => save({ edit_status: 'rejected', edit_note: note })}>Reject edit</button></div></form>
    <form onSubmit={event => { event.preventDefault(); save({ acceptable_actions: after, baseline_actions: before, criterion_evidence: reason }); }}><h4>2. Set acceptable actions independently</h4><fieldset><legend>Original image</legend>{choices(before, setBefore)}</fieldset><fieldset><legend>Edited image</legend>{choices(after, setAfter)}</fieldset><textarea aria-label="Expected action rationale" placeholder="Why are these actions acceptable for this task?" value={reason} maxLength={2000} onChange={event => setReason(event.target.value)} required/><button type="submit" disabled={busy || !reviewer.trim() || !reason.trim() || !before.length || !after.length}>Save expected actions & evaluate</button>{item.criterion_approved && <small>Expected actions approved and saved.</small>}</form></div>
    {message && <p role={failed ? 'alert' : 'status'}>{message}</p>}
  </section>;
}

function App() {
  const [run, setRun] = useState(null);
  const [status, setStatus] = useState(null);
  const [task, setTask] = useState(samples[0].task);
  const [backend, setBackend] = useState('muse');
  const [policyProvider, setPolicyProvider] = useState('baseten');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [selected, setSelected] = useState(0);
  const [error, setError] = useState('');
  const [working, setWorking] = useState(false);
  const [operation, setOperation] = useState('replay');
  const input = useRef(null);
  const timer = useRef(null);
  const running = working || run?.status === 'running';
  const current = run?.cases?.[selected];
  const complete = run?.status === 'completed';
  const prepared = run?.mode === 'prepared_example';
  const sameActionPass = current?.outcome === 'pass' && run?.original?.outcome === 'pass' && current?.observed_action === run?.original?.observed_action;

  async function loadSample(sample) {
    setError('');
    try {
      const response = await fetch(`/examples/${sample.image}`);
      if (!response.ok) throw new Error('Example image unavailable.');
      const blob = await response.blob();
      setFile(new File([blob], sample.image, { type: 'image/jpeg' }));
      setTask(sample.task); setBackend('muse'); setRun(null); setSelected(0);
    } catch (e) { setError(e.message); }
  }
  async function replay() {
    setFile(null); setTask(samples[0].task); setBackend('muse');
    setError(''); setOperation('replay'); setWorking(true); clearTimeout(timer.current);
    try { const data = await api('/api/replay', { method: 'POST' }); setRun(data); setTask(data.task); setBackend(data.backend); setSelected(0); }
    catch (e) { setError(e.message); }
    finally { setWorking(false); }
  }
  useEffect(() => {
    api('/api/status').then(setStatus).catch(e => setError(e.message));
    loadSample(samples[0]);
    return () => clearTimeout(timer.current);
  }, []);
  useEffect(() => {
    if (!file) { setPreview(null); return; }
    const url = URL.createObjectURL(file); setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => {
    if (run?.status !== 'running') return;
    let cancelled = false;
    const poll = async () => {
      try {
        const data = await api(`/api/runs/${run.id}`);
        if (!cancelled) { setRun(data); if (data.status === 'running') timer.current = setTimeout(poll, 1200); }
      } catch (e) { if (!cancelled) { setError(`Connection interrupted. ${e.message}`); timer.current = setTimeout(poll, 3000); } }
    };
    timer.current = setTimeout(poll, 800);
    return () => { cancelled = true; clearTimeout(timer.current); };
  }, [run?.id, run?.status]);

  async function live(event) {
    event.preventDefault(); setError(''); setOperation('live'); setWorking(true); setSelected(0);
    const form = new FormData(); form.append('task', task); form.append('backend', backend); form.append('policy_provider', policyProvider);
    try {
      if (file) form.append('image', file);
      else if (run?.original?.image_url) {
        const response = await fetch(run.original.image_url);
        if (!response.ok) throw new Error('Original image could not be loaded.');
        form.append('image', await response.blob(), 'original.png');
      } else throw new Error('Choose an input image first.');
      setRun(await api('/api/runs', { method: 'POST', body: form }));
    }
    catch (e) { setError(e.message); }
    finally { setWorking(false); }
  }
  async function retry() {
    setError(''); setOperation('live'); setWorking(true);
    try { setRun(await api(`/api/runs/${run.id}/retry`, { method: 'POST' })); }
    catch (e) { setError(e.message); }
    finally { setWorking(false); }
  }

  const originalImage = run?.original?.image_url || preview;
  return <div className="shell">
    <header className="topbar"><a className="brand" href="/" aria-label="Whatiff home"><span className="brand-mark"><i/><i/><i/><i/></span>whatiff<span className="brand-period">.</span></a></header>
    <main>
      <section className="hero"><div><h3>Intelligent Data Generation for Physical AI to Prevent Real-World Failures</h3><p>Create new scenarios from real robot observations, test how robots respond,  identify potential failures before they happen in the real world.</p></div></section>
      <details className="demo-settings"><summary>Change task, image or generator <span>Optional</span></summary>
      <form id="live-demo" className="setup" onSubmit={live}>
        <div className="task-input"><label htmlFor="task">01 / TASK INSTRUCTION</label><input id="task" value={task} maxLength={2000} onChange={e => setTask(e.target.value)} required disabled={running}/></div>
        <div className="scene-input"><label>02 / OBSERVATION</label><button type="button" className="upload" onClick={() => input.current.click()} disabled={running}><Icon name="upload" size={16}/><span>{file ? file.name : 'Synthetic tabletop'}<small>{file ? 'Uploaded scene · needs independent review' : 'Included example · or upload an image'}</small></span></button><input className="file-input" ref={input} type="file" aria-label="Upload tabletop observation" accept="image/png,image/jpeg,image/webp" onChange={e => { if (e.target.files[0]) setFile(e.target.files[0]); }} disabled={running}/>{file && <button className="reset-scene" type="button" onClick={() => { setFile(null); input.current.value = ''; }}>Use example scene</button>}</div>
        <div className="backend-input"><label htmlFor="backend">03 / GENERATOR</label><select id="backend" value={backend} onChange={e => setBackend(e.target.value)} disabled={running}><option value="controlled">SAM + Pillow · controlled</option><option value="muse">Muse Image · 3 edits</option></select></div>
        <div className="backend-input"><label htmlFor="policy-provider">04 / EVALUATOR POLICY</label><select id="policy-provider" value={policyProvider} onChange={e => setPolicyProvider(e.target.value)} disabled={running}><option value="baseten">Muse Spark Open · Baseten (default)</option><option value="meta">Hosted Muse Spark · explicit fallback</option></select></div>
      </form>

      </details>
      <div className="run-controls"><button className="run-button" type="submit" form="live-demo" disabled={running || !status?.meta_configured || (policyProvider === 'baseten' && !status?.baseten_configured) || (!file && !run?.original?.image_url)}>{running ? <span className="spinner"/> : <Icon name="play" size={17}/>} {running ? (operation === 'replay' ? 'Loading example…' : 'Running live…') : 'Run Experiments'}</button></div>
      {error && <div role="alert" className="error">{error}</div>}

      <ExecutionPanel execution={run?.execution}/>
      <section className="experiment" aria-labelledby="experiment-heading">
        <div className="experiment-bar"><div><span className={`mode-tag ${prepared ? '' : 'live'}`}>{prepared ? 'Examples' : run ? 'LIVE RUN' : 'Examples'}</span><h2 id="experiment-heading">{complete ? 'Compare the observation and decision.' : 'Start with an image'}</h2><p className="run-task">Task: “{run?.task || task}”</p></div><div className="experiment-actions"><button className="subtle-button" onClick={replay} disabled={running}><Icon name="repeat" size={16}/> Replay saved example<span className="zero-api">0 API calls</span></button>{complete && <a className="export-button" href={`/api/runs/${run.id}/report`} download><Icon name="download" size={16}/> Export JSON</a>}</div></div>
        <div className="sample-picker" aria-label="Kitchen examples">{samples.map(sample => <button type="button" key={sample.image} onClick={() => loadSample(sample)} disabled={running} aria-label={sample.name}><img src={`/examples/${sample.image}`} alt=""/><span>{sample.name}</span></button>)}</div>
        <a className="sample-credit" href="/examples/SOURCES.md">EPIC-KITCHENS · CC BY-NC 4.0</a>
        {prepared && <p className="recorded-note">Recorded live Meta decisions · Replay: 0 model calls</p>}
        {run?.status === 'running' && <div className="progress" role="status" aria-live="polite"><span className="spinner"/>Generating…</div>}
        {run?.status === 'failed' && <div className="error" role="alert"><div><strong>Run paused</strong><p>{run.phase} Reason: {run.error}. No successful result is claimed.</p></div><button className="subtle-button" onClick={retry} disabled={working}>Retry saved run <Icon name="repeat" size={15}/></button></div>}
        {run?.cases?.length > 1 && <div className="case-tabs" role="tablist" aria-label="Counterfactual intervention">{run.cases.map((item, index) => <button key={item.id} role="tab" aria-selected={selected === index} onClick={() => setSelected(index)} className={selected === index ? 'active' : ''}>{item.name}</button>)}</div>}

        <div className="comparison"><Observation kind="Original" task={run?.task || task} image={originalImage} data={run?.original} mode={run?.mode} loading={running} region={current?.checks?.rectangle_xyxy}/><div className="comparison-arrow"><Icon name="arrow" size={20}/></div><Observation kind="Counterfactual" image={current?.image_url} data={current} mode={run?.mode} loading={running} region={current?.checks?.rectangle_xyxy}/></div>
        {current?.checks?.rectangle_xyxy && <p className="annotation-note"><span className="annotation-swatch"/> Outline marks the intervention region. It is a display-only guide; the source images are unchanged.</p>}
        {current && <div className={`finding ${current.outcome}`}><div className="finding-heading"><span className="finding-icon"><Icon name={current.outcome === 'pass' ? 'check' : 'eye'} size={20}/></span><div><span className="small-label">ACTION-SELECTION EVALUATION</span><h3>{current.outcome === 'pass' ? 'The decision held up to this change.' : current.outcome === 'potential_failure' ? 'A potential action-selection failure.' : current.review?.edit_status === 'rejected' ? 'Edit rejected. Decision retained.' : 'Decision recorded. Review pending.'}</h3></div><Outcome value={current.outcome}/></div><p>{sameActionPass ? 'A valid invariance test: the same action remained appropriate after the change. No policy failure was found in this case.' : current.review?.edit_status === 'rejected' ? 'This edit is excluded from scoring. Its images and recorded decisions remain in the export.' : current.outcome === 'inconclusive' ? (!current.criterion_approved && current.validity.status !== 'valid' ? 'The image edit and expected action need review before this decision can be scored.' : !current.criterion_approved ? 'The expected action needs review before this decision can be scored.' : 'The image edit needs review before this decision can be scored.') : current.explanation} {current.outcome === 'pass' && 'This does not establish physical success or general robustness.'}</p></div>}

        {current && complete && <ReviewPanel key={`${run.id}-${current.id}`} run={run} item={current} onSaved={setRun}/>}
      </section>
      {complete && <div className="saved-files"><span>Generated Data</span><a href={run.artifact_directory ? `file://${run.artifact_directory}` : '#'} onClick={async event => { event.preventDefault(); try { await api(`/api/runs/${run.id}/reveal`, { method: 'POST' }); } catch (e) { setError(e.message); } }}>{run.artifact_directory || `data/web/${run.id}`}</a><small>Open in Finder</small></div>}
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<App/>);
