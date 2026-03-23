import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { fetchBackgroundTasks } from '../api';

const STATE_LABELS: Record<string, string> = {
  STARTED: 'Выполняется',
  SUCCESS: 'Успех',
  FAILURE: 'Ошибка',
  RETRY: 'Повтор',
  REVOKED: 'Отменена',
};

function shortTaskName(name: string): string {
  const parts = name.split('.');
  return parts.length > 0 ? parts[parts.length - 1] : name;
}

function TasksPage() {
  const { data: tasks = [], isLoading } = useQuery({
    queryKey: ['background-tasks'],
    queryFn: () => fetchBackgroundTasks(0, 200),
    refetchInterval: 5000,
  });

  if (isLoading) return <div className="page">Загрузка...</div>;

  return (
    <div className="page">
      <h1>Фоновые задачи</h1>
      <Link to="/games" className="btn">
        ← К списку игр
      </Link>

      <section style={{ marginTop: '1.5rem' }}>
        {tasks.length === 0 ? (
          <p>Нет записей о задачах</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Задача</th>
                <th>Статус</th>
                <th>Игра</th>
                <th>Документ</th>
                <th>Начало</th>
                <th>Окончание</th>
                <th>Ошибка / результат</th>
              </tr>
            </thead>
            <tbody>
              {tasks.map((j) => (
                <tr key={j.id}>
                  <td className="mono" title={j.task_name}>
                    {shortTaskName(j.task_name)}
                  </td>
                  <td>{STATE_LABELS[j.state] ?? j.state}</td>
                  <td>
                    {j.game_id != null && j.game_title ? (
                      <Link to={`/games/${j.game_id}/rules`}>{j.game_title}</Link>
                    ) : j.game_title ? (
                      j.game_title
                    ) : (
                      '—'
                    )}
                  </td>
                  <td className="mono">
                    {j.doc_id ? `${j.doc_id.slice(0, 12)}…` : '—'}
                  </td>
                  <td>{new Date(j.started_at).toLocaleString()}</td>
                  <td>
                    {j.finished_at ? new Date(j.finished_at).toLocaleString() : '—'}
                  </td>
                  <td style={{ maxWidth: '28rem', wordBreak: 'break-word' }}>
                    {j.error_message ?? (j.result_summary ? j.result_summary.slice(0, 200) : '—')}
                    {j.result_summary && j.result_summary.length > 200 ? '…' : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}

export default TasksPage;
