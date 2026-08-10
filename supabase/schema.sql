-- AtoCatch Supabase 스키마
-- 새 Supabase 프로젝트의 SQL Editor에서 이 파일을 그대로 실행하면 됩니다.
-- 필요한 시크릿(.env)은 README의 "실행 방법" 참고.

-- ============================================================
-- 1. RAG: rag_documents / match_rag_documents
--    rag_engine.py가 SUPABASE_SECRET_KEY(관리자 키, RLS 우회)로 접근.
--    app/data/*.pdf를 앱 첫 실행 시 자동으로 이 테이블에 임베딩·인덱싱함.
-- ============================================================

create extension if not exists vector;

create table if not exists rag_documents (
  id text primary key,
  doc text not null,
  embedding vector(1536) not null,  -- text-embedding-3-small 차원수
  source text not null,
  chunk_index int not null
);

-- 데이터가 적어(~500 chunk) 근사 인덱스(ivfflat)는 필요 없음 — 순차 스캔으로 충분

create or replace function match_rag_documents(
  query_embedding vector(1536),
  match_count int
)
returns table (
  id text,
  doc text,
  source text,
  chunk_index int,
  similarity float
)
language sql stable
as $$
  select
    id,
    doc,
    source,
    chunk_index,
    1 - (embedding <=> query_embedding) as similarity
  from rag_documents
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- ============================================================
-- 2. 로그인 / 분석 기록: analysis_history
--    app_main.py가 SUPABASE_PUBLISHABLE_KEY + 로그인한 사용자의 access token으로
--    접근. RLS 정책이 본인 user_id 행만 노출하도록 강제함.
--    사용자 인증 자체는 Supabase Auth(이메일/비밀번호)를 그대로 사용 — 별도 테이블 불필요.
-- ============================================================

create table if not exists analysis_history (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null,
  created_at timestamptz default now() not null,
  record_type text not null,        -- '설문조사' / '이미지분석'
  detail text,
  prediction jsonb,
  image_base64 text,
  gradcam_base64 text
);

alter table analysis_history enable row level security;

create policy "select_own_history"
  on analysis_history for select
  using (auth.uid() = user_id);

create policy "insert_own_history"
  on analysis_history for insert
  with check (auth.uid() = user_id);

create policy "delete_own_history"
  on analysis_history for delete
  using (auth.uid() = user_id);

-- ============================================================
-- 3. Supabase 대시보드에서 추가로 확인할 설정
--    (SQL로 제어되지 않는 프로젝트 레벨 설정)
-- ============================================================
-- Authentication → Sign In / Providers → Email
--   - Allow new users to sign up : ON
--   - Confirm email              : OFF (가입 직후 바로 로그인하는 현재 앱 흐름과 맞춤)
