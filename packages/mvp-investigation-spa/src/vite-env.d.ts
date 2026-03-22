/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_SENSEL_ORIGIN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
