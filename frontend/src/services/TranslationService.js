class TranslationService {
  static translateClassificationLabel(label, language = 'en') {
    const translations = {
      en: {
        1: "Direct Material - Production",
        2: "Indirect Material - Support", 
        3: "Tools & Equipment",
        4: "Consumable Items",
        5: "Other/Miscellaneous"
      },
      ja: {
        1: "直接材料 - 製造",
        2: "間接材料 - サポート",
        3: "工具・設備", 
        4: "消耗品",
        5: "その他・雑項目"
      }
    };
    
    return translations[language]?.[label] || translations.en[label] || `Label ${label}`;
  }

  static translateConfidenceLevel(level, language = 'en') {
    const translations = {
      en: { high: "High", medium: "Medium", low: "Low" },
      ja: { high: "高", medium: "中", low: "低" }
    };
    
    return translations[language]?.[level?.toLowerCase()] || level;
  }

  static translateMatchSource(source, language = 'en') {
    const translations = {
      en: {
        knowledge_base: "From Knowledge Base",
        supplier_bom: "From Supplier BOM", 
        hybrid: "Verified Match",
        no_match: "No Match"
      },
      ja: {
        knowledge_base: "知識ベースから",
        supplier_bom: "サプライヤーBOMから",
        hybrid: "検証済みマッチ", 
        no_match: "マッチなし"
      }
    };
    
    return translations[language]?.[source] || source;
  }

  static translateBoolean(value, language = 'en') {
    if (value == null) return '-';
    
    const translations = {
      en: { true: "Yes", false: "No" },
      ja: { true: "はい", false: "いいえ" }
    };
    
    return translations[language]?.[value.toString()] || value.toString();
  }

  static translateResultItem(item, language = 'en') {
    if (!item) return item;
    
    return {
      ...item,
      classification_description: this.translateClassificationLabel(item.qa_classification_label, language),
      confidence_level_text: this.translateConfidenceLevel(item.qa_confidence_level, language),
      match_source_text: this.translateMatchSource(item.match_source, language),
      action_path: this.translateActionPath(item.qa_classification_label, language),
      consumable_text: this.translateBoolean(item.consumable_jigs_tools, language)
    };
  }
}

export { TranslationService };
