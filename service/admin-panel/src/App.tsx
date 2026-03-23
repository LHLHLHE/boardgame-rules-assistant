import { BrowserRouter, Routes, Route, Link, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider, useAuth } from './auth';
import ProtectedLayout from './ProtectedLayout';
import DataManagement from './pages/DataManagement';
import GamesList from './pages/GamesList';
import GameForm from './pages/GameForm';
import TasksPage from './pages/TasksPage';
import UploadRules from './pages/UploadRules';
import Login from './pages/Login';
import UsersPage from './pages/UsersPage';
import './App.css';

const queryClient = new QueryClient();

function NavBar() {
  const { user, logout } = useAuth();
  if (!user) return null;
  return (
    <nav className="nav">
      <Link to="/">Игры</Link>
      <Link to="/games/new">Добавить игру</Link>
      <Link to="/tasks">Задачи</Link>
      {user.is_admin && (
        <>
          <Link to="/admin/data">Управление данными</Link>
          <Link to="/admin/users">Пользователи</Link>
        </>
      )}
      <span className="nav-user">
        {user.username}
        <button type="button" className="btn-link" onClick={() => logout()}>
          Выйти
        </button>
      </span>
    </nav>
  );
}

function AppShell() {
  return (
    <div className="app">
      <NavBar />
      <main>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route element={<ProtectedLayout />}>
            <Route path="/" element={<GamesList />} />
            <Route path="/games" element={<GamesList />} />
            <Route path="/games/new" element={<GameForm />} />
            <Route path="/games/:id/edit" element={<GameForm />} />
            <Route path="/games/:id/rules" element={<UploadRules />} />
            <Route path="/tasks" element={<TasksPage />} />
            <Route path="/admin/data" element={<DataManagement />} />
            <Route path="/admin/users" element={<UsersPage />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <AppShell />
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;
