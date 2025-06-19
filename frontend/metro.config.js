const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Add CSS support for web
config.resolver.assetExts.push('css');

module.exports = config;
