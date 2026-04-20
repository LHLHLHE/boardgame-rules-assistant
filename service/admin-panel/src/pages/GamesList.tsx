import { useState } from 'react';
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { deleteGame, fetchGames } from '../api';
import type { Game } from '../api';

const PAGE_SIZE_OPTIONS = [25, 50, 100] as const;

function GamesList() {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState('');
  const [pageState, setPageState] = useState(0);
  const [pageSize, setPageSize] = useState<number>(50);

  const { data, isFetching, error } = useQuery({
    queryKey: ['games', search, pageState, pageSize],
    queryFn: () => fetchGames(pageState * pageSize, pageSize, search),
    placeholderData: keepPreviousData,
  });

  const total = data?.total ?? 0;
  const list = data?.items ?? [];

  const maxPage = total > 0 ? Math.max(0, Math.ceil(total / pageSize) - 1) : 0;
  const clampedPage =
    data === undefined ? pageState : total === 0 ? 0 : Math.min(pageState, maxPage);

  if (data !== undefined && clampedPage !== pageState) {
    setPageState(clampedPage);
  }

  const skip = pageState * pageSize;

  const deleteMutation = useMutation({
    mutationFn: deleteGame,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['games'] });
    },
  });

  const empty = total === 0;
  const showInitialLoading = isFetching && data === undefined;
  const rangeFrom = total === 0 ? 0 : skip + 1;
  const rangeTo = Math.min(skip + list.length, total);
  const hasPrev = pageState > 0;
  const hasNext = skip + list.length < total;

  return (
    <div className="page">
      <h1>Игры</h1>
      <div className="form" style={{ marginBottom: '1rem', maxWidth: '24rem' }}>
        <label>
          Поиск по названию
          <input
            type="search"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPageState(0);
            }}
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
        <>
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
          <div
            className="form"
            style={{
              marginTop: '1rem',
              display: 'flex',
              flexWrap: 'wrap',
              alignItems: 'center',
              gap: '0.75rem 1rem',
            }}
          >
            <span>
              Показано {rangeFrom}–{rangeTo} из {total}
            </span>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', margin: 0 }}>
              На странице
              <select
                value={pageSize}
                onChange={(e) => {
                  setPageSize(Number(e.target.value));
                  setPageState(0);
                }}
              >
                {PAGE_SIZE_OPTIONS.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <button type="button" className="btn" disabled={!hasPrev} onClick={() => setPageState((p) => p - 1)}>
              Назад
            </button>
            <button type="button" className="btn" disabled={!hasNext} onClick={() => setPageState((p) => p + 1)}>
              Вперёд
            </button>
          </div>
        </>
      )}
    </div>
  );
}

export default GamesList;
