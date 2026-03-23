import { useState } from 'react';
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { deleteGame, fetchGames } from '../api';
import type { Game } from '../api';

function GamesList() {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState('');
  const { data: games, isFetching, error } = useQuery({
    queryKey: ['games', search],
    queryFn: () => fetchGames(0, 100, search),
    placeholderData: keepPreviousData,
  });

  const deleteMutation = useMutation({
    mutationFn: deleteGame,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['games'] });
    },
  });

  const list = games ?? [];
  const empty = !list.length;
  const showInitialLoading = isFetching && games === undefined;

  return (
    <div className="page">
      <h1>Игры</h1>
      <div className="form" style={{ marginBottom: '1rem', maxWidth: '24rem' }}>
        <label>
          Поиск по названию
          <input
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Подстрока в названии"
          />
        </label>
      </div>
      <Link to="/games/new" className="btn">+ Добавить игру</Link>
      {deleteMutation.isError && deleteMutation.error instanceof Error ? (
        <p className="error" style={{ marginTop: '1rem' }}>
          {deleteMutation.error.message}
        </p>
      ) : null}
      {error ? (
        <p className="error" style={{ marginTop: '1rem' }}>
          Ошибка загрузки
        </p>
      ) : showInitialLoading ? (
        <p style={{ marginTop: '1rem' }}>Загрузка...</p>
      ) : empty ? (
        <p style={{ marginTop: '1rem' }}>
          {search.trim() ? 'Ничего не найдено.' : 'Список пуст.'}
        </p>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Название</th>
              <th>Ссылка</th>
              <th>Действия</th>
            </tr>
          </thead>
          <tbody>
            {list.map((g: Game) => (
              <tr key={g.id}>
                <td>{g.id}</td>
                <td>{g.title}</td>
                <td>
                  {g.source_doc_url && <a href={g.source_doc_url} target="_blank" rel="noreferrer">Ссылка</a>}
                </td>
                <td>
                  <Link to={`/games/${g.id}/edit`}>Редактировать</Link>
                  {' | '}
                  <Link to={`/games/${g.id}/rules`}>Правила</Link>
                  {' | '}
                  <button
                    type="button"
                    className="danger"
                    disabled={deleteMutation.isPending}
                    onClick={() => {
                      if (
                        !window.confirm(
                          `Удалить игру «${g.title}» и связанные правила? Это действие нельзя отменить.`,
                        )
                      ) {
                        return;
                      }
                      deleteMutation.mutate(g.id);
                    }}
                  >
                    Удалить
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default GamesList;
