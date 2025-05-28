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
    if (item) {
        if (options?.type === "invitation") {
            if (!userData) item.id = `idTournament-${idNotify}`
            else item.id = `${userData.id}-idFriend`
        }
        else
            item.id = `idNotif-${idNotify}`
    }
    messageSpan.textContent = message;
    if (options?.type === "error") { // Red for error | Blue for invite
        item.classList.add("bg-red-600");
        acceptBtn.classList.add("hidden");
        rejectBtn.classList.add("hidden");
    } else if (options?.type === "invitation") {
        const idNotifyTournament = idNotify
        if (nameInvite && userData) nameInvite.innerText = `From ${userData.nickName}`;
        else if (nameInvite) nameInvite.innerText = `For the tournament`;
        item.classList.add("bg-gray-600");
        acceptBtn.classList.remove("hidden");
        rejectBtn.classList.remove("hidden");
        acceptBtn.onclick = async () => {
            if (options.onAccept) options.onAccept();
            if (userData) {
                hideNotification(0, userData.id);
                document.getElementById(`friendId-${userData.id}`)?.remove();
            }
            else {
                hideNotification(0, idNotifyTournament);
                document.getElementById(`idTournament-${idNotifyTournament}`)?.remove();
            }
        };
        rejectBtn.onclick = async () => {
            if (options.onRefuse) options.onRefuse();
            if (userData) {
                hideNotification(0, userData.id)
                document.getElementById(`friendId-${userData.id}`)?.remove()
            }
            else {
                hideNotification(0, idNotifyTournament)
                document.getElementById(`idTournament-${idNotifyTournament}`)?.remove()
            }
        };
    } else { // Default Green
        item.classList.remove("bg-red-600", "bg-blue-600");
        item.classList.add("bg-gray-800");
        acceptBtn.classList.add("hidden");
    }

    const notifys = notificationList.querySelectorAll("li");
    if (notifys && notifys.length >= 3)
        hideNotification(idNotify - 3);
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
        else {
            setTimeout(() => {
                item.classList.remove("translate-x-0");
                item.classList.add("translate-x-full");
                setTimeout(() => item.remove(), 500);
            }, 55000);
        }
    });

}

export function hideNotification(idNotify: number, idInvitation?: number) {
    console.log("hide notification", idNotify);
    if (idInvitation) {
        const item = document.getElementById(`${idInvitation}-idFriend`);
        if (!item)
            return;
        item.classList.remove("translate-x-0");
        item.classList.add("translate-x-full");
        setTimeout(() => item.remove(), 500);
        return ;
    }
    const item = document.getElementById(`idNotif-${idNotify}`);
    if (!item)
        return;
    item.classList.remove("translate-x-0");
    item.classList.add("translate-x-full");
    setTimeout(() => item.remove(), 500);
}
