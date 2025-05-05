
export function DispayNotification(message: string, options?: {
    type?: "error" | "invitation";
    onAccept?: () => void;
}) {
    const banner = document.getElementById("notificationBanner");
    const messageSpan = document.getElementById("notificationMessage");
    const actionBtn = document.getElementById("notificationActionAccept");
    if (!banner || !messageSpan || !actionBtn) {
        console.error("Notification elements missing");
        return;
    }
    messageSpan.textContent = message;
    if (options?.type === "error") { // Red for error | Blue for invite
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
    } else { // Default Green
        banner.classList.remove("bg-red-600", "bg-blue-600");
        banner.classList.add("bg-green-800");
        actionBtn.classList.add("hidden");
    }

    banner.classList.remove("hidden");
    window.requestAnimationFrame(() => {
        banner.classList.add("!right-0");
    });
    console.log('End of banner notification');

    requestAnimationFrame(() => {
        banner.classList.add("!right-0");
        setTimeout(() => {
            banner.classList.remove("!right-0");
            setTimeout(() => banner.classList.add("hidden"), 500);
        }, 5000);
    });

}

export function hideNotification() {
    const banner = document.getElementById("notificationBanner");
    if (!banner) return;
    banner.classList.remove("!right-0");
    setTimeout(() => banner.classList.add("hidden"), 500);
}