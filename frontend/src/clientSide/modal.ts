import { displayNotification } from "./notificationHandler.js";

let dropdownInitialized = false;
let selectedValue: string | null = null;

export function openModal() {
    const modal = document.getElementById("myModal") as HTMLDivElement | null;
    const app = document.getElementById("app") as HTMLDivElement | null;
    const modalContent = document.getElementById("modalContent") as HTMLDivElement | null;
    const submitBtn = document.getElementById("submitProspectEmail") as HTMLButtonElement | null;
    const closeBtn = document.getElementById("closeModalBtn") as HTMLButtonElement | null;
    const input = document.getElementById("emailProspect") as HTMLInputElement | null;

    if (!modal || !app || !modalContent || !submitBtn || !closeBtn || !input) {
        displayNotification("Erreur : HTML manquant, rechargez la page.");
        return;
    }

    // Affichage du modal avec animation
    modal.classList.remove("hidden", "opacity-0");
    modal.classList.add("flex", "opacity-100");

    requestAnimationFrame(() => {
        modalContent.classList.remove("opacity-0", "translate-y-full", "-translate-y-full");
        modalContent.classList.add("opacity-100", "translate-y-0");
        modal.classList.remove("backdrop-blur-0");
        modal.classList.add("backdrop-blur-sm");
    });

    // Initialiser dropdown une seule fois
    if (!dropdownInitialized) {
        initDropdown();
        dropdownInitialized = true;
    }

    // Bouton envoyer
    submitBtn.onclick = async () => {
        const ProspectEmail = input.value.trim();
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

        if (!selectedValue) {
            displayNotification("Merci de choisir votre type d'intérêt avant d'envoyer.");
            return;
        }

        if (!emailRegex.test(ProspectEmail)) {
            displayNotification("Veuillez entrer une adresse email valide.");
            return;
        }


        try {
            const res = await fetch("/api/add-contact", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email: ProspectEmail, type: selectedValue }),
            });

            const data = await res.json();

            if (res.ok) {
                displayNotification(`Merci ! Nous avons enregistré votre email : ${ProspectEmail}`);
                input.value = "";
                selectedValue = null;
                closeModal();
            } else if (res.status === 429) {
                displayNotification("Trop de tentatives, réessayez dans une minute !");
            } else {
                displayNotification(data.error || "Erreur lors de l'envoi.");
            }
        } catch (err) {
            console.error(err);
            displayNotification("Erreur de connexion au serveur.");
        }
    };

    // Bouton fermer
    closeBtn.onclick = () => {
        closeModal();
    };

    app.appendChild(modal);
}

function initDropdown() {
    const dropdownBtn = document.getElementById("dropdownBtn") as HTMLButtonElement;
    const dropdownMenu = document.getElementById("dropdownMenu") as HTMLUListElement;
    const selectedOption = document.getElementById("selectedOption") as HTMLSpanElement;

    if (!dropdownBtn || !dropdownMenu || !selectedOption) {
        console.warn("Dropdown introuvable");
        return;
    }

    dropdownBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        dropdownMenu.classList.toggle("hidden");
        dropdownBtn.querySelector("svg")?.classList.toggle("rotate-180");
    });

    dropdownMenu.querySelectorAll("li").forEach((li) => {
        li.addEventListener("click", () => {
            selectedValue = li.getAttribute("data-value");
            selectedOption.textContent = li.textContent || "Sélectionnez…";
            dropdownBtn.classList.remove("text-gray-500");
            dropdownMenu.classList.add("hidden");
            dropdownBtn.querySelector("svg")?.classList.remove("rotate-180");
        });
    });

    // Fermer le dropdown si clic à l'extérieur
    document.addEventListener("click", (e) => {
        if (!dropdownBtn.contains(e.target as Node) && !dropdownMenu.contains(e.target as Node)) {
            dropdownMenu.classList.add("hidden");
            dropdownBtn.querySelector("svg")?.classList.remove("rotate-180");
        }
    });
}

export function closeModal() {
    const modal = document.getElementById("myModal");
    const modalContent = document.getElementById("modalContent");
    if (!modal || !modalContent) return;

    modalContent.classList.remove("opacity-100", "translate-y-0");
    modalContent.classList.add("opacity-0", "-translate-y-full");

    modal.classList.remove("opacity-100");
    modal.classList.add("opacity-0");

    // Masquer après animation
    setTimeout(() => {
        modal.classList.remove("flex");
        modal.classList.add("hidden", "backdrop-blur-0");
        modal.classList.remove("opacity-0");
    }, 500);
}
