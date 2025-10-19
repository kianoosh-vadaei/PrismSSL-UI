const THEME_MAP = {
  dark: "ace/theme/dracula",
  light: "ace/theme/xcode",
};

function debounce(fn, wait = 250) {
  let handle;
  return function debounced(...args) {
    clearTimeout(handle);
    handle = setTimeout(() => fn.apply(this, args), wait);
  };
}

export function createAceEditor({ elementId, storageKey, defaultValue = "", mode = "python", onChange } = {}) {
  const el = document.getElementById(elementId);
  if (!el || typeof ace === "undefined") {
    console.warn("Ace editor not available", elementId);
    return null;
  }
  const initial = window.localStorage.getItem(storageKey) ?? defaultValue;
  const editor = ace.edit(elementId);
  editor.session.setMode(`ace/mode/${mode}`);
  editor.session.setUseWorker(false);
  editor.setOptions({
    highlightActiveLine: false,
    showPrintMargin: false,
    tabSize: 4,
    fontSize: "14px",
    useSoftTabs: true,
    enableBasicAutocompletion: true,
    enableLiveAutocompletion: true,
  });
  editor.setValue(initial, -1);
  const theme = themeForCurrentDocument();
  if (theme) {
    editor.setTheme(theme);
  }
  const persist = debounce((value) => {
    try {
      window.localStorage.setItem(storageKey, value);
    } catch (error) {
      console.warn("Unable to persist editor state", error);
    }
    if (onChange) {
      onChange(value);
    }
  }, 200);
  editor.session.on("change", () => {
    const value = editor.getValue();
    persist(value);
  });
  return editor;
}

function themeForCurrentDocument() {
  const theme = document.documentElement.getAttribute("data-theme");
  if (!theme) return null;
  return THEME_MAP[theme] || THEME_MAP.dark;
}

export function updateEditorTheme(editors, theme) {
  const aceTheme = THEME_MAP[theme] || THEME_MAP.dark;
  editors
    .filter(Boolean)
    .forEach((editor) => {
      editor.setTheme(aceTheme);
    });
}
