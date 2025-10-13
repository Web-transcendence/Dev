let idNotify: number = 0;



export function displayNotification(message: string, options?: {
    type?: "error";
    onAccept?: () => void;
    onRefuse?: () => void;
}) {
    const notificationList = document.getElementById("notificationList");
    const banner = document.getElementById("notificationBanner") as HTMLTemplateElement | null;
    if (!banner || !notificationList) {
        console.error("Notification elements missing");
        return;
    }
    const clone = banner.content.cloneNode(true) as HTMLElement | null;
    if (!clone) {
        console.error("Notification elements missing clone");
        return;
    }
    const messageSpan = clone.querySelector("#notificationMessage") as HTMLSpanElement | null;
    const acceptBtn = clone.querySelector("#notificationActionAccept") as HTMLButtonElement | null;
    const rejectBtn = clone.querySelector("#notificationActionReject") as HTMLButtonElement | null;
    const nameInvite = clone.querySelector("#nameInvite") as HTMLSpanElement | null;
    const item = clone.querySelector("li") as HTMLLIElement | null;
    if (!banner || !messageSpan || !acceptBtn || !rejectBtn || !item) {
        console.error("Notification elements missing");
        return;
    }
    if (item) item.id = `idNotif-${idNotify}`
    messageSpan.textContent = message;
    if (options?.type === "error") {
        item.classList.add("bg-red-600");
        acceptBtn.classList.add("hidden");
        rejectBtn.classList.add("hidden");
    } else { // Default Green
        item.classList.remove("bg-red-600", "bg-blue-600");
        item.classList.add("bg-fuchsia-200");
        acceptBtn.classList.add("hidden");
    }

    const notifys = notificationList.querySelectorAll("li");
    if (notifys && notifys.length >= 3) {
        hideNotification(idNotify - 3);
    }
    idNotify++;

    notificationList.appendChild(clone);
    window.requestAnimationFrame(() => {
        item.classList.remove("translate-x-full");
        item.classList.add("translate-x-0");
    });

    requestAnimationFrame(() => {
        if (options?.type !== "invitation") {
            setTimeout(() => {
                item.classList.remove("translate-x-0");
                item.classList.add("translate-x-full");
                setTimeout(() => item.remove(), 500);
            }, 5000);
        }
    });

}

export function hideNotification(idNotify: number) {
    const item = document.getElementById(`idNotif-${idNotify}`);
    if (!item) {
        return;
    }
    item.classList.remove("translate-x-0");
    item.classList.add("translate-x-full");
    setTimeout(() => item.remove(), 500);
}
