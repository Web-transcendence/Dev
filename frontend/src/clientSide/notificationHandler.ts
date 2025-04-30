
export function DispayNotification(message: string, options ?: {
    type?: "error" | "invitation",
        onAccept?: () => void
}) {
    const banner = document.getElementById("notificationBanner");
    const messageSpan = document.getElementById("notificationMessage");
    const actionBtn = document.getElementById("notificationAction");

    if (!banner || !messageSpan || !actionBtn) {
        console.error("Notification elements missing");
        return;
    }

    // Met à jour le message
    messageSpan.textContent = message;

    // Gère les styles selon le type
    if (options?.type === "error") {
        banner.classList.remove("bg-gray-800");
        banner.classList.add("bg-red-600");
        actionBtn.classList.add("hidden");
    } else if (options?.type === "invitation") {
        banner.classList.remove("bg-gray-800");
        banner.classList.add("bg-blue-600");
        actionBtn.classList.remove("hidden");
        actionBtn.textContent = "Accepter";
        actionBtn.onclick = () => {
            options.onAccept?.();
            hideNotification();
        };
    } else {
        // Par défaut
        banner.classList.remove("bg-red-600", "bg-blue-600");
        banner.classList.add("bg-gray-800");
        actionBtn.classList.add("hidden");
    }

    banner.classList.remove("hidden");
    window.requestAnimationFrame(() => {
        banner.classList.add("!right-0"); // glisse dedans
    });
    console.log('End of banner notification');

    requestAnimationFrame(() => {
        banner.classList.add("!right-0");

        // Retire la notification après un délai (ex: 5s visibles)
        setTimeout(() => {
            banner.classList.remove("!right-0");
            setTimeout(() => banner.classList.add("hidden"), 500); // après la transition
        }, 5000);
    });

}

export function hideNotification() {
    const banner = document.getElementById("notificationBanner");
    if (!banner) return;
    banner.classList.remove("!right-0");
    setTimeout(() => banner.classList.add("hidden"), 500);
}