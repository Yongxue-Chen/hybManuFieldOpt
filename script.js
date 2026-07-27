const navigationButton = document.querySelector(".nav-toggle");
const navigation = document.querySelector(".site-nav");

navigationButton?.addEventListener("click", () => {
  const isOpen = navigationButton.getAttribute("aria-expanded") === "true";
  navigationButton.setAttribute("aria-expanded", String(!isOpen));
  navigation?.classList.toggle("is-open", !isOpen);
});

navigation?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    navigationButton?.setAttribute("aria-expanded", "false");
    navigation.classList.remove("is-open");
  });
});

document.querySelectorAll("[data-copy-target]").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;

    try {
      await navigator.clipboard.writeText(target.textContent);
      const originalLabel = button.textContent;
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = originalLabel;
      }, 1600);
    } catch {
      button.textContent = "Select & copy";
    }
  });
});

document.querySelectorAll("[data-image]").forEach((slot) => {
  const image = new Image();
  image.alt = slot.dataset.alt || "";
  image.loading = "eager";
  image.decoding = "async";
  image.addEventListener("load", () => {
    slot.classList.add("has-image");
    slot.replaceChildren(image);
  });
  const separator = slot.dataset.image.includes("?") ? "&" : "?";
  image.src = `${slot.dataset.image}${separator}v=16`;
});

const year = document.getElementById("current-year");
if (year) year.textContent = new Date().getFullYear();
