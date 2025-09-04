import React from 'react';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { TranslationProvider } from './contexts/TranslationContext';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import UploadPage from './components/Upload';
import ProcessingPage from './components/ProcessingPage';
import ResultsPage from './components/Results';
import KnowledgeBasePage from './components/KnowledgeBase';
import SettingsPage from './components/SettingsPage';

function App() {
  return (
    <TranslationProvider>
      <HashRouter>
        <div className="App">
          <Layout>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/processing/:workflowId" element={<ProcessingPage />} />
              <Route path="/results/:workflowId" element={<ResultsPage />} />
              <Route path="/knowledge-base" element={<KnowledgeBasePage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </Layout>
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
            }}
          />
        </div>
      </HashRouter>
    </TranslationProvider>
  );
}

export default App;
