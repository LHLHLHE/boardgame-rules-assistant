const API_BASE = '/api/v1';

export function getAccessToken(): string | null {
  return localStorage.getItem('access_token');
}

export function setAccessToken(token: string | null): void {
  if (token) localStorage.setItem('access_token', token);
  else localStorage.removeItem('access_token');
}

function authHeaders(): Record<string, string> {
  const t = getAccessToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

/** Parses FastAPI error bodies: string detail or 422 array of { msg, ... }. */
export function errorMessageFromResponseBody(body: unknown): string {
  if (!body || typeof body !== 'object') return 'Request failed';
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const first = detail[0];
    if (first && typeof first === 'object' && first !== null && 'msg' in first) {
      const msg = (first as { msg: unknown }).msg;
      if (typeof msg === 'string') return msg;
    }
    if (typeof first === 'string') return first;
    try {
      return JSON.stringify(first);
    } catch {
      return 'Validation error';
    }
  }
  return 'Request failed';
}

export interface Me {
  id: number;
  username: string;
  email: string | null;
  is_admin: boolean;
  is_staff: boolean;
  created_at: string;
}

export async function login(username: string, password: string): Promise<{ access_token: string }> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const msg = errorMessageFromResponseBody(err);
    throw new Error(msg === 'Request failed' ? 'Login failed' : msg);
  }
  return res.json();
}

export async function fetchMe(): Promise<Me> {
  const res = await fetch(`${API_BASE}/auth/me`, { headers: { ...authHeaders() } });
  if (!res.ok) throw new Error('Not authenticated');
  return res.json();
}

export interface Game {
  id: number;
  title: string;
  source_doc_url: string | null;
  created_at: string;
  updated_at: string;
}

export async function fetchGames(skip = 0, limit = 100, search?: string): Promise<Game[]> {
  const params = new URLSearchParams({ skip: String(skip), limit: String(limit) });
  if (search !== undefined && search.trim() !== '') {
    params.set('search', search.trim());
  }
  const res = await fetch(`${API_BASE}/games?${params}`, {
    headers: { ...authHeaders() },
  });
  if (!res.ok) throw new Error('Failed to fetch games');
  return res.json();
}

export async function fetchGame(id: number): Promise<Game> {
  const res = await fetch(`${API_BASE}/games/${id}`, { headers: { ...authHeaders() } });
  if (!res.ok) throw new Error('Failed to fetch game');
  return res.json();
}

export async function createGame(data: { title: string; source_doc_url?: string }): Promise<Game> {
  const res = await fetch(`${API_BASE}/games`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to create game');
  return res.json();
}

export async function updateGame(id: number, data: Partial<{ title: string; source_doc_url: string }>): Promise<Game> {
  const res = await fetch(`${API_BASE}/games/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to update game');
  return res.json();
}

export async function deleteGame(gameId: number): Promise<void> {
  const res = await fetch(`${API_BASE}/games/${gameId}`, {
    method: 'DELETE',
    headers: { ...authHeaders() },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(errorMessageFromResponseBody(err));
  }
}

export interface RulesDocument {
  id: number;
  game_id: number;
  doc_id: string;
  storage_path: string;
  lang: string;
  status: string;
  created_at: string;
}

export async function fetchGameRules(gameId: number): Promise<RulesDocument[]> {
  const res = await fetch(`${API_BASE}/games/${gameId}/rules`, { headers: { ...authHeaders() } });
  if (!res.ok) throw new Error('Failed to fetch rules');
  return res.json();
}

export interface UploadRulesResponse {
  rules_document: RulesDocument;
  task_queued: boolean;
  tasks_url: string;
}

export async function uploadRules(
  gameId: number,
  file: File,
  lang = 'ru',
): Promise<UploadRulesResponse> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/games/${gameId}/rules?lang=${lang}`, {
    method: 'POST',
    body: form,
    headers: { ...authHeaders() },
  });
  if (!res.ok) {
    let message = 'Failed to upload rules';
    try {
      const errBody = (await res.json()) as { detail?: string };
      if (typeof errBody.detail === 'string') message = errBody.detail;
    } catch {
      /* ignore */
    }
    throw new Error(message);
  }
  return res.json();
}

export interface CreateGameWithRulesResponse {
  game: Game;
  rules_document: RulesDocument | null;
  task_queued: boolean;
  tasks_url: string;
}

export async function createGameWithRules(
  data: { title: string; source_doc_url?: string; lang?: string; file?: File },
): Promise<CreateGameWithRulesResponse> {
  const form = new FormData();
  form.append('title', data.title);
  if (data.source_doc_url) form.append('source_doc_url', data.source_doc_url);
  form.append('lang', data.lang ?? 'ru');
  if (data.file) form.append('file', data.file);
  const res = await fetch(`${API_BASE}/games/with-rules`, {
    method: 'POST',
    body: form,
    headers: { ...authHeaders() },
  });
  if (!res.ok) throw new Error('Failed to create game');
  return res.json();
}

export interface BackgroundTask {
  id: number;
  celery_task_id: string;
  task_name: string;
  state: string;
  started_at: string;
  finished_at: string | null;
  error_message: string | null;
  result_summary: string | null;
  related_entity_type: string | null;
  related_entity_id: number | null;
  game_title: string | null;
  game_id: number | null;
  doc_id: string | null;
}

export async function fetchBackgroundTasks(skip = 0, limit = 100): Promise<BackgroundTask[]> {
  const res = await fetch(`${API_BASE}/background-tasks?skip=${skip}&limit=${limit}`, {
    headers: { ...authHeaders() },
  });
  if (!res.ok) throw new Error('Failed to fetch background tasks');
  return res.json();
}

export interface InitializeResponse {
  status: string;
  games_created: number;
  rules_documents_created: number;
}

export async function initializeFromUpload(
  manifest: File,
  archive: File,
  options?: { limit?: number },
): Promise<InitializeResponse> {
  const form = new FormData();
  form.append('manifest', manifest);
  form.append('archive', archive);
  const limit = options?.limit;
  const qs =
    limit !== undefined && Number.isInteger(limit) && limit > 0 ? `?limit=${limit}` : '';
  const res = await fetch(`${API_BASE}/games/initialize${qs}`, {
    method: 'POST',
    body: form,
    headers: { ...authHeaders() },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const msg = errorMessageFromResponseBody(err);
    throw new Error(msg !== 'Request failed' ? msg : `Failed to initialize: ${res.status}`);
  }
  return res.json();
}

export interface ClearGamesResponse {
  status: string;
  games_deleted: number;
}

export async function clearGames(): Promise<ClearGamesResponse> {
  const res = await fetch(`${API_BASE}/games/clear-games`, {
    method: 'POST',
    headers: { ...authHeaders() },
  });
  if (!res.ok) throw new Error('Failed to clear games');
  return res.json();
}

export interface UserRow {
  id: number;
  username: string;
  email: string | null;
  is_admin: boolean;
  is_staff: boolean;
  created_at: string;
  updated_at: string;
}

export async function fetchUsers(): Promise<UserRow[]> {
  const res = await fetch(`${API_BASE}/users`, { headers: { ...authHeaders() } });
  if (!res.ok) throw new Error('Failed to fetch users');
  return res.json();
}

export async function createUser(data: {
  username: string;
  password: string;
  email?: string | null;
  is_admin: boolean;
  is_staff: boolean;
}): Promise<UserRow> {
  const res = await fetch(`${API_BASE}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(errorMessageFromResponseBody(err));
  }
  return res.json();
}

export async function updateUser(
  id: number,
  data: Partial<{ password: string; email: string | null; is_admin: boolean; is_staff: boolean }>,
): Promise<UserRow> {
  const res = await fetch(`${API_BASE}/users/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(errorMessageFromResponseBody(err));
  }
  return res.json();
}

export async function deleteUser(id: number): Promise<void> {
  const res = await fetch(`${API_BASE}/users/${id}`, {
    method: 'DELETE',
    headers: { ...authHeaders() },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(errorMessageFromResponseBody(err));
  }
}
