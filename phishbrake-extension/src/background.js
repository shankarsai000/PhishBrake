const MENU_ID = "phishbrake-scan-selection";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: MENU_ID,
    title: "Scan with PhishBrake",
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId !== MENU_ID || !info.selectionText?.trim()) {
    return;
  }

  await chrome.storage.local.set({ pendingScan: info.selectionText.trim() });
  if (chrome.action.openPopup) {
    await chrome.action.openPopup();
  }
});
