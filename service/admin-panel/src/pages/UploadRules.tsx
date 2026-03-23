import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useState, useRef, useMemo } from 'react';
import { fetchGame, fetchGameRules, uploadRules } from '../api';
import type { RulesDocument } from '../api';

const STATUS_LABELS: Record<string, string> = {
  pending: 'Ожидает',
  processing: 'Обрабатывается',
  indexed: 'Проиндексировано',
  failed: 'Ошибка',
};

function UploadRules() {
  const { id } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [taskQueuedMessage, setTaskQueuedMessage] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const { data: game, isLoading: gameLoading } = useQuery({
    queryKey: ['game', id],
    queryFn: () => fetchGame(Number(id)),
    enabled: Boolean(id),
  });

  const { data: rules, isLoading: rulesLoading } = useQuery({
    queryKey: ['rules', id],
    queryFn: () => fetchGameRules(Number(id!)),
    enabled: Boolean(id),
  });

  const uploadBlocked = useMemo(
    () =>
      Boolean(
        rules?.some((r) => r.status === 'pending' || r.status === 'processing'),
      ),
    [rules],
  );

  const uploadMutation = useMutation({
    mutationFn: ({ file }: { file: File }) => uploadRules(Number(id!), file),
    onSuccess: (data) => {
      setUploadError(null);
      queryClient.invalidateQueries({ queryKey: ['rules', id] });
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      if (data.task_queued) {
        setTaskQueuedMessage('Задача поставлена в очередь');
      }
    },
    onError: (err: Error) => {
      setUploadError(err.message);
    },
  });

  const handleFile = (file: File) => {
    if (uploadBlocked) return;
    const lower = file.name.toLowerCase();
    if (!lower.endsWith('.txt') && !lower.endsWith('.pdf')) {
      alert('Загружайте только .txt или .pdf файлы с правилами');
      return;
    }
    uploadMutation.mutate({ file });
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (uploadBlocked) return;
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = '';
  };

  if (gameLoading || !game) return <div className="page">Загрузка...</div>;

  return (
    <div className="page">
      <h1>Правила: {game.title}</h1>
      <button onClick={() => navigate('/games')}>← К списку игр</button>

      {taskQueuedMessage && (
        <p className="message success">
          {taskQueuedMessage}. <Link to="/tasks">Отследить статус</Link>
        </p>
      )}

      {uploadError && (
        <p className="error" style={{ marginTop: '0.75rem' }}>
          {uploadError}
        </p>
      )}

      <section className="upload-section">
        <h2>Загрузить правила</h2>
        {uploadBlocked && (
          <p className="message" style={{ marginBottom: '0.75rem' }}>
            Индексация ещё не завершена (статус «Ожидает» или «Обрабатывается»). Дождитесь
            «Проиндексировано» или «Ошибка», либо смотрите{' '}
            <Link to="/tasks">задачи</Link>.
          </p>
        )}
        <div
          className={`dropzone ${dragOver ? 'drag-over' : ''} ${uploadBlocked ? 'dropzone-disabled' : ''}`}
          style={
            uploadBlocked
              ? { opacity: 0.55, cursor: 'not-allowed', pointerEvents: 'none' }
              : undefined
          }
          onDragOver={
            uploadBlocked
              ? undefined
              : (e) => {
                  e.preventDefault();
                  setDragOver(true);
                }
          }
          onDragLeave={uploadBlocked ? undefined : () => setDragOver(false)}
          onDrop={uploadBlocked ? undefined : handleDrop}
          onClick={
            uploadBlocked ? undefined : () => fileInputRef.current?.click()
          }
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.pdf"
            onChange={handleChange}
            disabled={uploadBlocked}
            style={{ display: 'none' }}
          />
          {uploadMutation.isPending
            ? 'Загрузка...'
            : uploadBlocked
              ? 'Загрузка недоступна до завершения индексации'
              : 'Перетащите .txt или .pdf сюда или кликните для выбора'}
        </div>
      </section>

      <section>
        <h2>Загруженные документы</h2>
        {rulesLoading ? (
          <p>Загрузка...</p>
        ) : !rules?.length ? (
          <p>Нет загруженных правил</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>doc_id</th>
                <th>Статус</th>
                <th>Создан</th>
              </tr>
            </thead>
            <tbody>
              {rules.map((r: RulesDocument) => (
                <tr key={r.id}>
                  <td>{r.id}</td>
                  <td className="mono">{r.doc_id.slice(0, 12)}...</td>
                  <td>{STATUS_LABELS[r.status] ?? r.status}</td>
                  <td>{new Date(r.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}

export default UploadRules;
