import {UserData} from "./user.js";

let idNotify: number = 0;

export function displayNotification(message: string, options?: {
    type?: "error" | "invitation";
    onAccept?: () => void;
    onRefuse?: () => void;
}, userData?: UserData) {
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
    if (options?.type === "error") { // Red for error | Blue for invite
        item.classList.add("bg-red-600");
        acceptBtn.classList.add("hidden");
        rejectBtn.classList.add("hidden");
    } else if (options?.type === "invitation") {
        if (userData && nameInvite) nameInvite.innerText = `From ${userData.nickName}`;
        const idOfInvite = idNotify;
        item.classList.add("bg-blue-600");
        acceptBtn.classList.remove("hidden");
        rejectBtn.classList.remove("hidden");
        acceptBtn.onclick = async () => {
            if (options.onAccept) options.onAccept();
            hideNotification(idOfInvite);
        };
        rejectBtn.onclick = async () => {
            if (options.onRefuse) options.onRefuse();
            hideNotification(idOfInvite);
        };
    } else { // Default Green
        item.classList.remove("bg-red-600", "bg-blue-600");
        item.classList.add("bg-green-800");
        acceptBtn.classList.add("hidden");
    }

    const notifys = notificationList.querySelectorAll("li");
    console.log('notifys', notifys);
    if (notifys && notifys.length >= 3) {
        console.log("Notification to remove", notifys.length - 4);
        hideNotification(idNotify - 3);
    }
    idNotify++;

    notificationList.appendChild(clone);
    window.requestAnimationFrame(() => {
        item.classList.remove("translate-x-full");
        item.classList.add("translate-x-0");
    });
    console.log('End of item notification');

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
    console.log("hide notification", idNotify);
    const item = document.getElementById(`idNotif-${idNotify}`);
    if (!item) {
        console.log('NOT FOUND BITA')
        return;
    }
    item.classList.remove("translate-x-0");
    item.classList.add("translate-x-full");
    setTimeout(() => item.remove(), 500);
}
