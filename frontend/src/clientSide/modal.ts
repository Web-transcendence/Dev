import {displayNotification} from "./notificationHandler.js";

export function openModal() {
    const modal = document.getElementById("myModal") as HTMLDivElement | undefined;
    const app = document.getElementById("app") as HTMLDivElement | undefined;
    const modalContent = document.getElementById("modalContent") as HTMLDivElement | undefined;
    const submitBtn = document.getElementById("submitProspectEmail") as HTMLButtonElement | null;
    const closeBtn = document.getElementById("closeModalBtn") as HTMLButtonElement | null;
    const input = document.getElementById("emailProspect") as HTMLInputElement | null;

    if (!modal || !app || !modalContent || !submitBtn || !closeBtn || !input) {
        displayNotification('Missing Html refresh page');
        return;
    }

    // Affiche le modal
    modal.classList.remove("hidden");
    modal.classList.add("flex");
    window.requestAnimationFrame(() => {
        modalContent.classList.remove("opacity-0", "translate-y-full", "-translate-y-full");
        modalContent.classList.add("opacity-100", "translate-y-0");
        modal.classList.remove("backdrop-blur-0");
        modal.classList.add("backdrop-blur-sm");
    });

    const dropdownBtn = document.getElementById("dropdownBtn") as HTMLButtonElement;
    const dropdownMenu = document.getElementById("dropdownMenu") as HTMLUListElement;
    const selectedOption = document.getElementById("selectedOption") as HTMLSpanElement;

    let selectedValue: string | null = null;

    dropdownBtn.addEventListener("click", () => {
        dropdownMenu.classList.toggle("hidden");
        dropdownBtn.querySelector("svg")?.classList.toggle("rotate-180");
    });

    dropdownMenu.querySelectorAll("li").forEach((li) => {
        li.addEventListener("click", () => {
            selectedValue = li.getAttribute("data-value");
            selectedOption.textContent = li.textContent;
            dropdownBtn.classList.remove("text-gray-500");
            dropdownMenu.classList.add("hidden");
            dropdownBtn.querySelector("svg")?.classList.remove("rotate-180");
        });
    });

// Optionnel : fermer le dropdown si clic à l'extérieur
    document.addEventListener("click", (e) => {
        if (!dropdownBtn.contains(e.target as Node) && !dropdownMenu.contains(e.target as Node)) {
            dropdownMenu.classList.add("hidden");
            dropdownBtn.querySelector("svg")?.classList.remove("rotate-180");
        }
    });


    submitBtn.onclick = async () => {
        const ProspectEmail = input.value.trim();
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

        if (emailRegex.test(ProspectEmail)) {
            try {
                const res = await fetch("/api/add-contact", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ email: ProspectEmail }),
                });

                const data = await res.json();
                if (res.ok) {
                    displayNotification(`Merci ! Nous avons enregistré votre email : ${ProspectEmail}`);
                    input.value = "";
                    closeModal();
                } else if (res.status === 429) {
                    displayNotification("Trop de tentatives, réessayez dans une minute !");
                } else {
                    displayNotification(`${data.error}`);
                }
            } catch (err) {
                displayNotification(`Impossible de contacter le serveur.`);
                console.error(err);
            }
        } else {
            displayNotification("Veuillez entrer une adresse email valide.");
        }
    };

    closeBtn.onclick = () => {
        closeModal();
    };

    app.appendChild(modal);
}

export function closeModal() {
    const modal = document.getElementById("myModal");
    const modalContent = document.getElementById("modalContent");
    if (!modal || !modalContent) return;

    modalContent.classList.remove("opacity-100", "translate-y-0");
    modalContent.classList.add("opacity-0", "-translate-y-full");
    modal.classList.remove("backdrop-blur-sm");
    modal.classList.add("backdrop-blur-0");

    setTimeout(() => {
        modal.classList.remove("flex");
        modal.classList.add("hidden");

        modalContent.classList.remove("-translate-y-full");
        modalContent.classList.add("translate-y-full");
    }, 500);
}
