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

const year = document.getElementById("current-year");
if (year) year.textContent = new Date().getFullYear();
