import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useState, useRef } from 'react';
import { fetchGame, createGame, createGameWithRules, updateGame } from '../api';

type GameForForm = { title: string; source_doc_url?: string };

function GameFormFields({
  gameId,
  initialGame,
  isEdit,
  onSuccess,
}: {
  gameId: number | undefined;
  initialGame: GameForForm | null | undefined;
  isEdit: boolean;
  onSuccess: (msg?: string) => void;
}) {
  const queryClient = useQueryClient();
  const [title, setTitle] = useState(initialGame?.title ?? '');
  const [sourceDocUrl, setSourceDocUrl] = useState(
    initialGame?.source_doc_url ?? ''
  );
  const [rulesFile, setRulesFile] = useState<File | null>(null);
  const [taskQueuedMessage, setTaskQueuedMessage] = useState<string | null>(
    null
  );
  const fileInputRef = useRef<HTMLInputElement>(null);

  const createMutation = useMutation({
    mutationFn: createGame,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['games'] });
      onSuccess();
    },
  });

  const createWithRulesMutation = useMutation({
    mutationFn: createGameWithRules,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['games'] });
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      if (data.task_queued) {
        setTaskQueuedMessage('Задача поставлена в очередь');
        setTimeout(() => onSuccess(), 2000);
      } else {
        onSuccess();
      }
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: number; data: { title?: string; source_doc_url?: string } }) =>
      updateGame(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['games'] });
      onSuccess();
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isEdit && gameId) {
      updateMutation.mutate({
        id: gameId,
        data: { title, source_doc_url: sourceDocUrl || undefined },
      });
    } else if (rulesFile) {
      createWithRulesMutation.mutate({
        title,
        source_doc_url: sourceDocUrl || undefined,
        file: rulesFile,
      });
    } else {
      createMutation.mutate({
        title,
        source_doc_url: sourceDocUrl || undefined,
      });
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && (file.name.endsWith('.txt') || file.name.toLowerCase().endsWith('.pdf'))) {
      setRulesFile(file);
    } else if (file) {
      setRulesFile(null);
      alert('Поддерживаются только .txt и .pdf');
    } else {
      setRulesFile(null);
    }
    e.target.value = '';
  };

  return (
    <>
      <h1>{isEdit ? 'Редактировать игру' : 'Добавить игру'}</h1>
      {taskQueuedMessage && (
        <p className="message success">
          {taskQueuedMessage}. <Link to="/tasks">Отследить статус</Link>
        </p>
      )}
      <form onSubmit={handleSubmit} className="form">
        <label>
          Название
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            required
          />
        </label>
        <label>
          Ссылка на источник
          <input
            type="url"
            value={sourceDocUrl}
            onChange={(e) => setSourceDocUrl(e.target.value)}
            placeholder="https://..."
          />
        </label>
        {!isEdit && (
          <label>
            Файл правил (опционально)
            <div
              className={`dropzone mini ${rulesFile ? 'has-file' : ''}`}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.pdf"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
              {rulesFile ? rulesFile.name : 'Перетащите .txt или .pdf или кликните'}
            </div>
          </label>
        )}
        <button
          type="submit"
          disabled={
            createMutation.isPending ||
            createWithRulesMutation.isPending ||
            updateMutation.isPending
          }
        >
          {isEdit ? 'Сохранить' : 'Создать'}
        </button>
      </form>
    </>
  );
}

function GameForm() {
  const { id } = useParams();
  const navigate = useNavigate();
  const isEdit = Boolean(id);
  const { data: game, isLoading } = useQuery({
    queryKey: ['game', id],
    queryFn: () => fetchGame(Number(id)),
    enabled: isEdit,
  });

  if (isEdit && isLoading) return <div className="page">Загрузка...</div>;

  const formKey = isEdit ? (game?.id ?? 'loading') : 'new';
  const gameForForm = game as GameForForm | undefined;

  return (
    <div className="page">
      <GameFormFields
        key={formKey}
        gameId={id ? Number(id) : undefined}
        initialGame={gameForForm}
        isEdit={isEdit}
        onSuccess={() => navigate('/games')}
      />
    </div>
  );
}

export default GameForm;
