-- WSmart-Route cloud runner control plane (CR-04, #124).
-- Used by logic/controllers/cloud/control/supabase.py. Apply with `supabase db push`
-- or psql. Single-tenant: any authenticated user of the project may use the runner;
-- the service role bypasses row-level security.

create table if not exists wsr_jobs (
    job_id      text primary key,
    created_at  double precision not null,
    rev         bigint not null default 0,
    doc         jsonb not null
);

create table if not exists wsr_tasks (
    task_id        text primary key,
    job_id         text not null references wsr_jobs (job_id) on delete cascade,
    seq            bigint not null,
    state          text not null,
    lease_expires  double precision,
    rev            bigint not null default 0,
    doc            jsonb not null
);
create index if not exists wsr_tasks_queue on wsr_tasks (state, seq);
create index if not exists wsr_tasks_job on wsr_tasks (job_id, seq);
create index if not exists wsr_tasks_leases on wsr_tasks (state, lease_expires);

create table if not exists wsr_events (
    id      bigserial primary key,
    job_id  text not null references wsr_jobs (job_id) on delete cascade,
    at      double precision not null,
    doc     jsonb not null
);
create index if not exists wsr_events_job on wsr_events (job_id, id);

create table if not exists wsr_metrics (
    id       bigserial primary key,
    task_id  text not null references wsr_tasks (task_id) on delete cascade,
    at       double precision not null,
    doc      jsonb not null
);
create index if not exists wsr_metrics_task on wsr_metrics (task_id, id);

create table if not exists wsr_results (
    task_id  text not null references wsr_tasks (task_id) on delete cascade,
    relpath  text not null,
    doc      jsonb not null,
    primary key (task_id, relpath)
);

-- A job and its tasks in one transaction. p_tasks is a JSON array of task documents.
create or replace function wsr_insert_job(p_job jsonb, p_tasks jsonb)
returns void language plpgsql as $$
begin
    insert into wsr_jobs (job_id, created_at, doc)
    values (p_job->>'job_id', (p_job->>'created_at')::double precision, p_job);
    insert into wsr_tasks (task_id, job_id, seq, state, lease_expires, rev, doc)
    select t->>'task_id', t->>'job_id', (t->>'seq')::bigint, t->>'state',
           (t->>'lease_expires')::double precision, coalesce((t->>'rev')::bigint, 0), t
    from jsonb_array_elements(p_tasks) as t;
end $$;

-- Lease the oldest queued task (of p_job_id if given); concurrent callers skip locked rows,
-- so two callers never receive the same task. Returns the task document, or null.
create or replace function wsr_lease_task(p_owner text, p_lease_expires double precision, p_job_id text default null)
returns jsonb language plpgsql as $$
declare
    v_id  text;
    v_doc jsonb;
begin
    select task_id into v_id
    from wsr_tasks
    where state = 'queued' and (p_job_id is null or job_id = p_job_id)
    order by seq
    limit 1
    for update skip locked;
    if v_id is null then
        return null;
    end if;
    update wsr_tasks
    set state = 'placing',
        lease_expires = p_lease_expires,
        rev = rev + 1,
        doc = doc || jsonb_build_object(
            'state', 'placing', 'lease_owner', p_owner, 'lease_expires', p_lease_expires, 'rev', rev + 1)
    where task_id = v_id
    returning doc into v_doc;
    return v_doc;
end $$;

-- Requeue every placing task whose lease ended before p_now. Returns a JSON array of task ids.
create or replace function wsr_expire_leases(p_now double precision)
returns jsonb language sql as $$
    with expired as (
        update wsr_tasks
        set state = 'queued',
            lease_expires = null,
            rev = rev + 1,
            doc = doc || jsonb_build_object('state', 'queued', 'lease_owner', null, 'lease_expires', null, 'rev', rev + 1)
        where state = 'placing' and lease_expires < p_now
        returning task_id
    )
    select coalesce(jsonb_agg(task_id order by task_id), '[]'::jsonb) from expired;
$$;

-- Replace a task's results index. p_docs is a JSON array of result documents.
create or replace function wsr_replace_results(p_task_id text, p_docs jsonb)
returns void language plpgsql as $$
begin
    delete from wsr_results where task_id = p_task_id;
    insert into wsr_results (task_id, relpath, doc)
    select p_task_id, d->>'relpath', d from jsonb_array_elements(p_docs) as d;
end $$;

-- Row-level security: authenticated users of the project; the service role bypasses it.
alter table wsr_jobs enable row level security;
alter table wsr_tasks enable row level security;
alter table wsr_events enable row level security;
alter table wsr_metrics enable row level security;
alter table wsr_results enable row level security;

do $$
declare
    t text;
begin
    foreach t in array array['wsr_jobs', 'wsr_tasks', 'wsr_events', 'wsr_metrics', 'wsr_results'] loop
        if exists (select 1 from pg_roles where rolname = 'authenticated') then
            execute format('drop policy if exists %I on %I', t || '_authenticated', t);
            execute format(
                'create policy %I on %I for all to authenticated using (true) with check (true)',
                t || '_authenticated', t);
        end if;
    end loop;
end $$;

do $$
begin
    if exists (select 1 from pg_roles where rolname = 'authenticated') then
        grant execute on function wsr_insert_job(jsonb, jsonb) to authenticated;
        grant execute on function wsr_lease_task(text, double precision, text) to authenticated;
        grant execute on function wsr_expire_leases(double precision) to authenticated;
        grant execute on function wsr_replace_results(text, jsonb) to authenticated;
    end if;
    -- Live updates for the Studio (Supabase Realtime).
    if exists (select 1 from pg_publication where pubname = 'supabase_realtime') then
        alter publication supabase_realtime add table wsr_tasks, wsr_events;
    end if;
end $$;
