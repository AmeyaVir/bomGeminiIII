import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Globe } from 'lucide-react';
import { useTranslation } from '../contexts/TranslationContext';
import Card from '../components/ui/Card';
import LanguageSelector from './LanguageSelector';

function SettingsPage() {
  const navigate = useNavigate();
  const { t } = useTranslation();

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto p-6">
        <div className="mb-8">
          <button
            onClick={() => navigate('/dashboard')}
            className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-4"
          >
            <ArrowLeft className="h-4 w-4 mr-1" />
            {t('common.back')} {t('common.to')} {t('navigation.dashboard')}
          </button>
          <h1 className="text-3xl font-bold text-gray-900">{t('settings.title')}</h1>
        </div>

        <Card className="p-6">
          <div className="flex items-center mb-4">
            <Globe className="h-5 w-5 text-gray-400 mr-2" />
            <h2 className="text-lg font-medium text-gray-900">{t('settings.language')}</h2>
          </div>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {t('settings.resultsLanguage')}
              </label>
              <p className="text-sm text-gray-500 mb-3">
                {t('settings.resultsLanguageDescription')}
              </p>
              <LanguageSelector />
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

export default SettingsPage;
