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

    // ⚡️ Assignation directe (empêche les doublons)
    submitBtn.onclick = () => {
        const ProspectEmail = input.value.trim();
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

        if (emailRegex.test(ProspectEmail)) {
            displayNotification(`Merci ! Nous avons enregistré votre email : ${ProspectEmail}`);
            input.value = "";
            closeModal();
        } else {
            displayNotification("❌ Veuillez entrer une adresse email valide.");
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
