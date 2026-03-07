document.addEventListener("DOMContentLoaded", function () {
  // Case 1: links with .model-copy class
  document.querySelectorAll("a.model-copy").forEach(function (link) {
    addCopyButton(link, link.textContent.trim());
  });

  // Case 2: headings with .model-copy class — find the code element inside
  document.querySelectorAll("h1.model-copy, h2.model-copy, h3.model-copy, h4.model-copy, h5.model-copy, h6.model-copy").forEach(function (heading) {
    const code = heading.querySelector("code");
    if (code) addCopyButton(code, code.textContent.trim());
  });
});

function addCopyButton(target, text) {
  const btn = document.createElement("button");
  btn.className = "copy-model-btn";
  btn.title = "Copy model name";
  btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
  btn.addEventListener("click", function (e) {
    e.preventDefault();
    navigator.clipboard.writeText(text).then(() => {
      btn.classList.add("copied");
      btn.title = "Copied!";
      setTimeout(() => {
        btn.classList.remove("copied");
        btn.title = "Copy model name";
      }, 1500);
    });
  });
  target.insertAdjacentElement("afterend", btn);
}