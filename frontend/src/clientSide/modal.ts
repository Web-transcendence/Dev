
import {displayNotification} from "./notificationHandler.js";
import {Pong} from "./pong.js";
import {TowerDefense} from "./td.js";
import {loadPart} from "./insert.js";
import {fetchInvitation} from "./invitation.js";

export async function openModal(nickname: string, id: number): Promise<void> {
    const modal = document.getElementById("myModal") as HTMLDivElement | undefined;
    const app = document.getElementById("app") as HTMLDivElement | undefined;
    const modalContent = document.getElementById("modalContent") as HTMLDivElement | undefined;
    const nameFriend = document.getElementById("nameFriend") as HTMLDivElement | undefined;
    if (!modal || !nameFriend || !app || !modalContent) {
        displayNotification('Missing Html refresh page')
        return ;
    }
    modal.classList.remove("hidden");
    modal.classList.add("flex");
    window.requestAnimationFrame(() => {
        modalContent.classList.remove("opacity-0", "translate-y-full", "-translate-y-full");
        modalContent.classList.add("opacity-100", "translate-y-0");
        modal.classList.remove("backdrop-blur-0");
        modal.classList.add("backdrop-blur-sm");
    });

    nameFriend.innerHTML = `Which game you want to play with ${nickname} ?`;
    document.getElementById('invitePong')?.addEventListener("click", async () => {
        const roomId = await fetchInvitation('match-server', id);
        displayNotification('Invitation send to ${nickname} !')
        await loadPart('/pongRemote')
        Pong('remote', roomId)
    });
    document.getElementById('inviteTowerDefense')?.addEventListener("click", async () => {
        const roomId = await fetchInvitation('tower-defense', id);
        displayNotification('Invitation send to ${nickname} !')
        await loadPart('/towerRemote')
        TowerDefense(roomId)
    });
    modal.addEventListener("click", () => {
        closeModal()
    });
    document.getElementById("closeModalBtn")?.addEventListener("click", () => {
        closeModal()
    });
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