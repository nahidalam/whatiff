import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './browser-tests', timeout: 240000, workers: 1,
  use: { baseURL: 'http://127.0.0.1:8790', headless: true,
    launchOptions: process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {} },
  outputDir: 'data/browser-tests', reporter: 'list',
});
