const scrollButton = document.querySelector(".scroll-to-top");
const copyButton = document.querySelector(".copy-bibtex-btn");
const bibtex = document.querySelector("#bibtex-code");

window.addEventListener(
  "scroll",
  () => scrollButton?.classList.toggle("visible", window.scrollY > 300),
  { passive: true }
);

scrollButton?.addEventListener("click", () => {
  window.scrollTo({ top: 0, behavior: "smooth" });
});

copyButton?.addEventListener("click", async () => {
  if (!bibtex) return;

  try {
    await navigator.clipboard.writeText(bibtex.textContent);
  } catch {
    const textArea = document.createElement("textarea");
    textArea.value = bibtex.textContent;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand("copy");
    textArea.remove();
  }

  const label = copyButton.querySelector("span");
  copyButton.classList.add("copied");
  if (label) label.textContent = "Copied";
  window.setTimeout(() => {
    copyButton.classList.remove("copied");
    if (label) label.textContent = "Copy";
  }, 1800);
});
