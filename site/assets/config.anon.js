// Reviewer build. scripts/build.py swaps this in for config.js in the anonymous site.
// Nothing here may identify the authors; the build fails if it does.
window.MLB_CONFIG = {
  anonymous: true,
  title: "MusicListenBench",
  lab: "Anonymous authors (under review)",
  labUrl: null,
  githubRepo: "https://anonymous.4open.science/r/MusicListenBench-F38B",
  csvSourceUrl: "https://anonymous.4open.science/r/MusicListenBench-F38B/leaderboard.csv",
  datasetUrl: null,  // the item files are in the code repository (data/); the About page links to it
  spaceUrl: null,
  paperUrl: null,
  contact: null,
  benchmarkVersion: "1.0",
  itemsPerAspect: 250,
  csvPath: "leaderboard.csv",
};
