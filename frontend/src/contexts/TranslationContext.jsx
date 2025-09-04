import React, { createContext, useContext, useState } from 'react';

const TranslationContext = createContext();

const translations = {
  en: {
    common: {
      loading: "Loading...",
      error: "Error",
      success: "Success",
      cancel: "Cancel",
      confirm: "Confirm",
      save: "Save",
      delete: "Delete",
      edit: "Edit",
      view: "View",
      download: "Download",
      upload: "Upload",
      search: "Search",
      filter: "Filter",
      export: "Export",
      back: "Back",
      next: "Next",
      previous: "Previous",
      close: "Close",
      yes: "Yes",
      no: "No",
      clear: "Clear",
      actions: "Actions",
      created: "Created",
      to: "to"
    },
    navigation: {
      dashboard: "Dashboard",
      upload: "Upload",
      processing: "Processing",
      results: "Results"
    },
    results: {
      title: "BOM Comparison Results",
      workflowId: "Workflow ID",
      withItemClassificationReasons: "With Item Classification & Reasons",
      backToDashboard: "Back to Dashboard",
      materialsProcessed: "Materials Processed",
      successfulMatches: "Successful Matches",
      averageConfidence: "Average Confidence",
      knowledgeBaseMatches: "Knowledge Base Matches",
      exportResults: "Export Results",
      resultsExported: "Results exported successfully!",
      failedToLoadResults: "The results for this workflow could not be loaded.",
      noMaterialsMatch: "No materials match the selected filter criteria.",
      noReasonProvided: "No reason provided",
      columns: {
        sno: "S.No.",
        materialName: "Material Name",
        qcProcess: "QC Process/WI Step",
        consumable: "Consumable/Jigs/Tools",
        partNumber: "Part Number",
        qty: "Qty",
        uom: "UoM",
        vendor: "Vendor",
        classification: "Classification",
        confidence: "Confidence",
        actionPath: "Action Path",
        supplierMatch: "Supplier Match",
        reason: "Reason"
      }
    },
    knowledgeBase: {
      title: "Knowledge Base",
      subtitle: "Database of previously processed items for enhanced matching accuracy",
      totalItems: "Total Items",
      totalWorkflows: "Total Workflows",
      totalMatches: "Total Matches",
      matchRate: "Match Rate",
      searchItems: "Search items in knowledge base...",
      noItems: "No items found",
      noItemsDescription: "No items match your search criteria.",
      pendingApprovals: "Pending Approvals",
      review: "Review"
    },
    dashboard: {
      title: "Dashboard",
      subtitle: "Monitor your autonomous BOM processing workflows with QA classification",
      showing: "Showing",
      of: "of",
      startProcessing: "Start Processing"
    },
    settings: {
      title: "Settings",
      language: "Language Settings",
      resultsLanguage: "Results Display Language",
      resultsLanguageDescription: "Choose the language for displaying BOM comparison results",
      knowledgeBase: "Knowledge Base Settings",
      enableKnowledgeBase: "Enable Knowledge Base",
      knowledgeBaseDescription: "Use historical data to improve matching accuracy",
      clearKnowledgeBase: "Clear Knowledge Base",
      clearKnowledgeBaseDescription: "Remove all stored historical data",
      confirmClear: "Are you sure you want to clear all knowledge base data?"
    }
  },
  ja: {
    common: {
      loading: "読み込み中...",
      error: "エラー",
      success: "成功",
      cancel: "キャンセル",
      confirm: "確認",
      save: "保存",
      delete: "削除",
      edit: "編集",
      view: "表示",
      download: "ダウンロード",
      upload: "アップロード",
      search: "検索",
      filter: "フィルター",
      export: "エクスポート",
      back: "戻る",
      next: "次へ",
      previous: "前へ",
      close: "閉じる",
      yes: "はい",
      no: "いいえ",
      clear: "クリア",
      actions: "操作",
      created: "作成日",
      to: "から"
    },
    navigation: {
      dashboard: "ダッシュボード",
      upload: "アップロード",
      processing: "処理中",
      results: "結果"
    },
    results: {
      title: "BOM比較結果",
      workflowId: "ワークフローID",
      withItemClassificationReasons: "アイテム分類と理由付き",
      backToDashboard: "ダッシュボードに戻る",
      materialsProcessed: "処理済み材料",
      successfulMatches: "成功マッチ",
      averageConfidence: "平均信頼度",
      knowledgeBaseMatches: "知識ベースマッチ",
      exportResults: "結果エクスポート",
      resultsExported: "結果が正常にエクスポートされました！",
      failedToLoadResults: "このワークフローの結果を読み込めませんでした。",
      noMaterialsMatch: "選択されたフィルター条件に一致する材料はありません。",
      noReasonProvided: "理由が提供されていません",
      columns: {
        sno: "番号",
        materialName: "材料名",
        qcProcess: "QCプロセス/WIステップ",
        consumable: "消耗品/治具/工具",
        partNumber: "部品番号",
        qty: "数量",
        uom: "単位",
        vendor: "ベンダー",
        classification: "分類",
        confidence: "信頼度",
        actionPath: "アクションパス",
        supplierMatch: "サプライヤーマッチ",
        reason: "理由"
      }
    },
    knowledgeBase: {
      title: "知識ベース",
      subtitle: "Database of previously processed items for enhanced matching accuracy",
      totalItems: "総アイテム数",
      totalWorkflows: "総ワークフロー数",
      totalMatches: "総マッチ数",
      matchRate: "マッチ率",
      searchItems: "知識ベース内のアイテムを検索...",
      noItems: "アイテムが見つかりません",
      noItemsDescription: "検索条件に一致するアイテムはありません。",
      pendingApprovals: "承認待ち",
      review: "確認"
    },
    dashboard: {
      title: "ダッシュボード",
      subtitle: "QA分類機能付きの自律的BOM処理ワークフローを監視",
      showing: "表示中",
      of: "/",
      startProcessing: "処理開始"
    },
    settings: {
      title: "設定",
      language: "言語設定",
      resultsLanguage: "結果表示言語",
      resultsLanguageDescription: "BOM比較結果の表示言語を選択してください",
      knowledgeBase: "知識ベース設定",
      enableKnowledgeBase: "知識ベースを有効化",
      knowledgeBaseDescription: "過去のデータを使用してマッチング精度を向上させる",
      clearKnowledgeBase: "知識ベースをクリア",
      clearKnowledgeBaseDescription: "保存されたすべての過去データを削除",
      confirmClear: "すべての知識ベースデータをクリアしてもよろしいですか？"
    }
  }
};

export function TranslationProvider({ children }) {
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const [loading, setLoading] = useState(false);

  const contextValue = {
    currentLanguage,
    changeLanguage: (lang) => {
      setCurrentLanguage(lang);
    },
    t: (key, fallback = key) => {
      const keys = key.split('.');
      let value = translations[currentLanguage];
      for (const k of keys) {
        value = value?.[k];
        if (value === undefined) {
          console.warn(`Translation key not found: ${key} for language: ${currentLanguage}`);
          value = translations.en;
          for (const k2 of keys) {
            value = value?.[k2];
            if (value === undefined) {
              console.warn(`Default translation key not found: ${key}`);
              return fallback;
            }
          }
          break;
        }
      }
      return value || fallback;
    },
    loading,
    availableLanguages: [
      { code: 'en', name: 'English', nativeName: 'English' },
      { code: 'ja', name: 'Japanese', nativeName: '日本語' }
    ]
  };

  return (
    <TranslationContext.Provider value={contextValue}>
      {children}
    </TranslationContext.Provider>
  );
}

export function useTranslation() {
  const context = useContext(TranslationContext);
  if (context === undefined) {
    throw new Error('useTranslation must be used within a TranslationProvider');
  }
  return context;
}
