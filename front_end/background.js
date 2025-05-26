// background.js
console.log("🔧 background.js loaded");

chrome.action.onClicked.addListener((tab) => {
  console.log("🔧 Icon clicked; injecting content.js into tab", tab.id);
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ["content.js"]
  })
  .then(() => console.log("🔧 content.js injected"))
  .catch(err => console.error("🔧 Injection failed:", err));
});
