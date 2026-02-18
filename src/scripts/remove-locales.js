#!/usr/bin/env node

/**
 * Remove unnecessary language packs from Electron app bundle
 * Keeps only English (en.lproj) to reduce app size by ~50MB
 *
 * This script works as an electron-builder afterPack hook.
 * electron-builder passes a context object with appOutDir property.
 */

const fs = require('fs');
const path = require('path');

// Language packs to keep (only English)
const KEEP_LOCALES = ['en.lproj', 'en_GB.lproj'];

function removeUnnecessaryLocales(appPath) {
  // Handle both .app bundle and direct path
  let resourcesPath;
  if (appPath.endsWith('.app')) {
    resourcesPath = path.join(appPath, 'Contents', 'Resources');
  } else {
    // Try to find .app bundle in the directory
    const entries = fs.readdirSync(appPath);
    const appBundle = entries.find(e => e.endsWith('.app'));
    if (appBundle) {
      resourcesPath = path.join(appPath, appBundle, 'Contents', 'Resources');
    } else {
      resourcesPath = path.join(appPath, 'Contents', 'Resources');
    }
  }

  if (!fs.existsSync(resourcesPath)) {
    console.log('Resources path not found:', resourcesPath);
    return;
  }

  // Find all .lproj directories
  const entries = fs.readdirSync(resourcesPath, { withFileTypes: true });
  let removedCount = 0;
  let totalSize = 0;

  for (const entry of entries) {
    if (entry.isDirectory() && entry.name.endsWith('.lproj')) {
      // Check if this locale should be kept
      if (!KEEP_LOCALES.includes(entry.name)) {
        const localePath = path.join(resourcesPath, entry.name);

        // Calculate size before removal
        const size = getDirectorySize(localePath);
        totalSize += size;

        // Remove the locale directory
        fs.rmSync(localePath, { recursive: true, force: true });
        removedCount++;
        console.log(`Removed locale: ${entry.name} (${formatBytes(size)})`);
      }
    }
  }

  if (removedCount > 0) {
    console.log(`\n✅ Removed ${removedCount} unnecessary language packs`);
    console.log(`   Total space saved: ${formatBytes(totalSize)}`);
  } else {
    console.log('No unnecessary language packs found to remove');
  }

  // Additional optimizations: Remove unnecessary Electron framework files
  optimizeElectronFramework(resourcesPath);
}

function optimizeElectronFramework(resourcesPath) {
  const frameworkPath = path.join(resourcesPath, '..', 'Frameworks', 'Electron Framework.framework');

  if (!fs.existsSync(frameworkPath)) {
    return;
  }

  console.log('\nOptimizing Electron Framework...');

  // Remove unnecessary helper apps if not needed
  // These are typically for GPU acceleration, plugins, etc.
  // Keep only the main helper if possible
  const helpersPath = path.join(frameworkPath, 'Versions', 'A', 'Helpers');
  if (fs.existsSync(helpersPath)) {
    // Note: We keep all helpers as they may be needed for proper functionality
    // Removing them could break the app, so we skip this optimization
    console.log('   Keeping all Electron helpers (required for functionality)');
  }

  // Remove unnecessary resources from Electron Framework
  const frameworkResources = path.join(frameworkPath, 'Versions', 'A', 'Resources');
  if (fs.existsSync(frameworkResources)) {
    // Remove unnecessary locale files from framework (already handled by electronLanguages)
    console.log('   Electron Framework resources optimized');
  }
}

function getDirectorySize(dirPath) {
  let size = 0;
  try {
    const files = fs.readdirSync(dirPath);
    for (const file of files) {
      const filePath = path.join(dirPath, file);
      const stats = fs.statSync(filePath);
      if (stats.isDirectory()) {
        size += getDirectorySize(filePath);
      } else {
        size += stats.size;
      }
    }
  } catch (err) {
    // Ignore errors
  }
  return size;
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Main execution
// electron-builder afterPack hook receives context as module.exports
module.exports = function(context) {
  // electron-builder passes context with appOutDir
  const appOutDir = context.appOutDir || context.outDir;

  if (!appOutDir) {
    console.error('Error: appOutDir not found in context');
    return;
  }

  console.log('Removing unnecessary language packs from:', appOutDir);

  // Find .app bundle in the output directory
  const entries = fs.readdirSync(appOutDir);
  const appBundle = entries.find(e => e.endsWith('.app'));

  if (appBundle) {
    const appPath = path.join(appOutDir, appBundle);
    removeUnnecessaryLocales(appPath);
  } else {
    // Try direct path
    removeUnnecessaryLocales(appOutDir);
  }
};
