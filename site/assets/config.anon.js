// Reviewer build. scripts/build.py swaps this in for config.js in the anonymous site.
// Nothing here may identify the authors; the build fails if it does.
window.MLB_CONFIG = {
  anonymous: true,
  title: "MusicListenBench",
  lab: "Anonymous authors (under review)",
  labUrl: null,
  githubRepo: null,  // TODO: Anonymous GitHub mirror, e.g. "https://anonymous.4open.science/r/XXXX"
  csvSourceUrl: null, // optional: the CSV inside the mirror, e.g. ".../r/XXXX/leaderboard.csv"
  datasetUrl: null,  // TODO: anonymous dataset location, or leave null if it is in the supplementary material
  spaceUrl: null,
  paperUrl: null,
  contact: null,
  benchmarkVersion: "1.0",
  itemsPerAspect: 500,
  csvPath: "leaderboard.csv",
};
