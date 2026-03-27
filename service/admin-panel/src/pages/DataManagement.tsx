import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../auth';
import { fetchGames, initializeFromUpload, clearGames } from '../api';

function DataManagement() {
  const { user: current } = useAuth();
  const queryClient = useQueryClient();
  const [manifestFile, setManifestFile] = useState<File | null>(null);
  const [archiveFile, setArchiveFile] = useState<File | null>(null);
  const [limitInput, setLimitInput] = useState('');
  const [message, setMessage] = useState<{ type: 'ok' | 'err'; text: string } | null>(null);

  const { data: gamesData, isLoading } = useQuery({
    queryKey: ['games'],
    queryFn: () => fetchGames(0, 1),
  });

  const hasGames = (gamesData?.total ?? 0) > 0;

  const parsedLimit = limitInput.trim() === '' ? undefined : Number.parseInt(limitInput, 10);
  const limitValid =
    limitInput.trim() === '' || (Number.isInteger(parsedLimit) && parsedLimit !== undefined && parsedLimit > 0);
  const limitForRequest =
    limitValid && parsedLimit !== undefined && parsedLimit > 0 ? parsedLimit : undefined;

  const initAllowed =
    Boolean(manifestFile && archiveFile && limitValid) && (!hasGames || limitForRequest !== undefined);

  const initMutation = useMutation({
    mutationFn: () => {
      if (!manifestFile || !archiveFile) throw new Error('Выберите манифест и архив');
      if (hasGames && limitForRequest === undefined) {
        throw new Error('Укажите максимум новых игр (целое число > 0), если в БД уже есть игры');
      }
      return initializeFromUpload(manifestFile, archiveFile, {
        limit: limitForRequest,
      });
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['games'] });
      setMessage({
        type: 'ok',
        text: `Загружено: ${data.games_created} игр, ${data.rules_documents_created} документов`,
      });
      setManifestFile(null);
      setArchiveFile(null);
    },
    onError: (err: Error) => {
      setMessage({ type: 'err', text: err.message });
    },
  });

  const clearMutation = useMutation({
    mutationFn: clearGames,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['games'] });
      setMessage({ type: 'ok', text: `Удалено игр: ${data.games_deleted}` });
    },
    onError: (err: Error) => {
      setMessage({ type: 'err', text: err.message });
    },
  });

  const handleInit = () => {
    setMessage(null);
    initMutation.mutate();
  };

  const handleClear = () => {
    if (!window.confirm('Удалить все игры и их правила? Это действие нельзя отменить.')) return;
    setMessage(null);
    clearMutation.mutate();
  };

  if (!current?.is_admin) {
    return <Navigate to="/" replace />;
  }

  if (isLoading) return <div className="page">Загрузка...</div>;

  return (
    <div className="page">
      <h1>Управление данными</h1>

      <section className="upload-section">
        <h2>Инициализация из манифеста и архива</h2>
        <p>
          Загрузите манифест (CSV) и архив (ZIP) с текстами правил. При пустой БД можно загрузить все строки
          манифеста или ограничить число <strong>новых</strong> игр за этот запуск (порядок строк в CSV важен).
          Если в БД уже есть игры, укажите лимит — будет добавлена порция новых игр (без полной очистки).
        </p>
        <div className="form">
          <label>
            Манифест (index_manifest.csv)
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setManifestFile(e.target.files?.[0] ?? null)}
            />
          </label>
          <label>
            Архив с data/rules_texts_cleaned_good/*.txt
            <input
              type="file"
              accept=".zip"
              onChange={(e) => setArchiveFile(e.target.files?.[0] ?? null)}
            />
          </label>
          <label>
            Макс. новых игр (необязательно)
            <input
              type="number"
              min={1}
              step={1}
              placeholder={hasGames ? 'обязательно при непустой БД' : 'пусто = все строки'}
              value={limitInput}
              onChange={(e) => setLimitInput(e.target.value)}
            />
          </label>
          <button
            onClick={handleInit}
            disabled={!initAllowed || initMutation.isPending}
          >
            {initMutation.isPending ? 'Загрузка...' : 'Загрузить'}
          </button>
          {hasGames && !limitForRequest && (
            <p className="hint">Укажите максимум новых игр, чтобы добавить данные без очистки БД.</p>
          )}
          {!limitValid && limitInput.trim() !== '' && (
            <p className="hint">Введите целое число больше 0 или оставьте поле пустым.</p>
          )}
        </div>
      </section>

      <section>
        <h2>Очистка данных</h2>
        <p>Удалить все игры и их правила. Используйте для повторной полной инициализации.</p>
        <button
          onClick={handleClear}
          disabled={!hasGames || clearMutation.isPending}
          className="danger"
        >
          {clearMutation.isPending ? 'Удаление...' : 'Очистить все данные'}
        </button>
      </section>

      {message && (
        <p className={message.type === 'ok' ? 'msg-ok' : 'msg-err'}>{message.text}</p>
      )}
    </div>
  );
}

export default DataManagement;
