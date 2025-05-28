
import {displayNotification} from "./notificationHandler.js";
import {Pong} from "./pong.js";
import {TowerDefense} from "./td.js";
import {loadPart} from "./insert.js";
import {fetchInvitation} from "./invitation.js";
import {navigate} from './front.js'

let currentInviteId: number | null = null;
let currentNickname: string | null = null;

export function setupModalListeners() {
    const invitePongBtn = document.getElementById('invitePong');
    const inviteTowerBtn = document.getElementById('inviteTowerDefense');
    const closeBtn = document.getElementById("closeModalBtn");
    const modal = document.getElementById("myModal");

    invitePongBtn?.addEventListener("click", async () => {
        if (currentInviteId !== null && currentNickname !== null) {
            const roomId = await fetchInvitation('match-server', currentInviteId);
            displayNotification(`Invitation sent to ${currentNickname} !`);
            await navigate('/pongFriend');
            Pong('remote', roomId);
            closeModal();
        }
    });

    inviteTowerBtn?.addEventListener("click", async () => {
        if (currentInviteId !== null && currentNickname !== null) {
            const roomId = await fetchInvitation('tower-defense', currentInviteId);
            displayNotification(`Invitation sent to ${currentNickname} !`);
            await loadPart('/towerFriend');
            TowerDefense(roomId);
            closeModal();
        }
    });

    modal?.addEventListener("click", (event) => {
        if (event.target === modal) closeModal();
    });

    closeBtn?.addEventListener("click", () => {
        closeModal();
    });
}

export async function openModal(nickname: string, id: number): Promise<void> {
    const modal = document.getElementById("myModal") as HTMLDivElement;
    const modalContent = document.getElementById("modalContent") as HTMLDivElement;
    const nameFriend = document.getElementById("nameFriend") as HTMLSpanElement;

    if (!modal || !modalContent || !nameFriend) {
        displayNotification('Missing HTML — refresh page');
        return;
    }

    currentNickname = nickname;
    currentInviteId = id;

    nameFriend.innerText = `Which game do you want to play with ${nickname}?`;

    modal.classList.remove("hidden");
    modal.classList.add("flex");

    requestAnimationFrame(() => {
        modalContent.classList.remove("opacity-0", "translate-y-full", "-translate-y-full");
        modalContent.classList.add("opacity-100", "translate-y-0");
        modal.classList.remove("backdrop-blur-0");
        modal.classList.add("backdrop-blur-sm");
    });
}

// export async function openModal(nickname: string, id: number): Promise<void> {
//     const modal = document.getElementById("myModal") as HTMLDivElement | undefined;
//     const modalContent = document.getElementById("modalContent") as HTMLDivElement | undefined;
//     const nameFriend = document.getElementById("nameFriend") as HTMLDivElement | undefined;
//     if (!modal || !nameFriend || !modalContent) {
//         displayNotification('Missing Html refresh page')
//         return ;
//     }
//     modal.classList.remove("hidden");
//     modal.classList.add("flex");
//     window.requestAnimationFrame(() => {
//         modalContent.classList.remove("opacity-0", "translate-y-full", "-translate-y-full");
//         modalContent.classList.add("opacity-100", "translate-y-0");
//         modal.classList.remove("backdrop-blur-0");
//         modal.classList.add("backdrop-blur-sm");
//     });
//
//     nameFriend.innerHTML = `Which game you want to play with ${nickname} ?`;
//     document.getElementById('invitePong')?.addEventListener("click", async () => {
//         const roomId = await fetchInvitation('match-server', id);
//         displayNotification(`Invitation send to ${nickname} !`)
//         await loadPart('/GameFriend')
//         Pong('remote', roomId)
//     });
//     document.getElementById('inviteTowerDefense')?.addEventListener("click", async () => {
//         const roomId = await fetchInvitation('tower-defense', id);
//         displayNotification(`Invitation send to ${nickname} !`)
//         await loadPart('/GameFriend')
//         TowerDefense(roomId)
//     });
//     modal.addEventListener("click", () => {
//         closeModal()
//     });
//     document.getElementById("closeModalBtn")?.addEventListener("click", () => {
//         closeModal()
//     });
// }

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
    console.log('closeModal FINISHED');
}