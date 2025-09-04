import React from 'react';
import { Globe } from 'lucide-react';
import { useTranslation } from '../contexts/TranslationContext';

function LanguageSelector({ className = "" }) {
  const { currentLanguage, changeLanguage, availableLanguages } = useTranslation();

  return (
    <div className={`relative inline-block ${className}`}>
      <select
        value={currentLanguage}
        onChange={(e) => changeLanguage(e.target.value)}
        className="appearance-none bg-white border border-gray-300 rounded-md py-2 pl-3 pr-8 text-sm leading-5 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
      >
        {availableLanguages.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.nativeName}
          </option>
        ))}
      </select>
      <div className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
        <Globe className="h-4 w-4 text-gray-400" />
      </div>
    </div>
  );
}

export default LanguageSelector;
