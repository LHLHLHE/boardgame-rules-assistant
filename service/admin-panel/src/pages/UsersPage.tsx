import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Link, Navigate } from 'react-router-dom';
import { useAuth } from '../auth';
import { createUser, deleteUser, fetchUsers, updateUser, type UserRow } from '../api';

function UsersPage() {
  const { user: current } = useAuth();
  const queryClient = useQueryClient();
  const [message, setMessage] = useState<{ type: 'ok' | 'err'; text: string } | null>(null);

  const { data: users = [], isLoading, error } = useQuery({
    queryKey: ['users'],
    queryFn: fetchUsers,
  });

  const createMutation = useMutation({
    mutationFn: createUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
      setMessage({ type: 'ok', text: 'Пользователь создан' });
    },
    onError: (err: Error) => setMessage({ type: 'err', text: err.message }),
  });

  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [newIsAdmin, setNewIsAdmin] = useState(false);
  const [newIsStaff, setNewIsStaff] = useState(true);

  if (!current?.is_admin) {
    return <Navigate to="/" replace />;
  }

  if (isLoading) return <div className="page">Загрузка…</div>;
  if (error) return <div className="page msg-err">{(error as Error).message}</div>;

  return (
    <div className="page">
      <h1>Пользователи</h1>
      <p>
        <Link to="/">← К играм</Link>
      </p>

      <section className="upload-section">
        <h2>Новый пользователь</h2>
        <div className="form">
          <label>
            Логин
            <input value={newUsername} onChange={(e) => setNewUsername(e.target.value)} />
          </label>
          <label>
            Пароль (мин. 8 символов)
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
            />
          </label>
          <label>
            <input
              type="checkbox"
              checked={newIsAdmin}
              onChange={(e) => setNewIsAdmin(e.target.checked)}
            />{' '}
            Администратор
          </label>
          <label>
            <input
              type="checkbox"
              checked={newIsStaff}
              onChange={(e) => setNewIsStaff(e.target.checked)}
            />{' '}
            Модератор (staff)
          </label>
          <button
            type="button"
            disabled={createMutation.isPending || !newUsername || newPassword.length < 8}
            onClick={() =>
              createMutation.mutate({
                username: newUsername,
                password: newPassword,
                is_admin: newIsAdmin,
                is_staff: newIsStaff,
              })
            }
          >
            {createMutation.isPending ? 'Создание…' : 'Создать'}
          </button>
        </div>
      </section>

      <table className="table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Логин</th>
            <th>Роли</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {users.map((u: UserRow) => (
            <UserRowEditor key={u.id} u={u} currentId={current.id} />
          ))}
        </tbody>
      </table>

      {message && (
        <p className={message.type === 'ok' ? 'msg-ok' : 'msg-err'}>{message.text}</p>
      )}
    </div>
  );
}

function UserRowEditor({ u, currentId }: { u: UserRow; currentId: number }) {
  const queryClient = useQueryClient();
  const [isAdmin, setIsAdmin] = useState(u.is_admin);
  const [isStaff, setIsStaff] = useState(u.is_staff);

  const updateMut = useMutation({
    mutationFn: () => updateUser(u.id, { is_admin: isAdmin, is_staff: isStaff }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['users'] }),
  });

  const deleteMut = useMutation({
    mutationFn: () => deleteUser(u.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['users'] }),
  });

  const canDelete = u.id !== currentId;

  return (
    <tr>
      <td>{u.id}</td>
      <td>{u.username}</td>
      <td>
        <label>
          <input
            type="checkbox"
            checked={isAdmin}
            onChange={(e) => setIsAdmin(e.target.checked)}
          />{' '}
          admin
        </label>{' '}
        <label>
          <input
            type="checkbox"
            checked={isStaff}
            onChange={(e) => setIsStaff(e.target.checked)}
          />{' '}
          staff
        </label>
        <button type="button" disabled={updateMut.isPending} onClick={() => updateMut.mutate()}>
          Сохранить
        </button>
      </td>
      <td>
        {canDelete && (
          <button
            type="button"
            className="danger"
            disabled={deleteMut.isPending}
            onClick={() => {
              if (window.confirm(`Удалить ${u.username}?`)) deleteMut.mutate();
            }}
          >
            Удалить
          </button>
        )}
      </td>
    </tr>
  );
}

export default UsersPage;
