import {addFriend, fetchUserInformation, FriendIds, getFriendList, removeFriend, UserData} from "./user.js";

export async function friendList() {
    try {
        const friendIds: FriendIds = await getFriendList();
        type FriendKey = keyof FriendIds;
        const sections: { key: FriendKey; listId: string; templateId: string }[] = [
            { key: "pendingIds", listId: "requestList", templateId: "requestTemplate" },
            { key: "acceptedIds", listId: "acceptedList", templateId: "acceptedTemplate" },
            { key: "receivedIds", listId: "receivedList", templateId: "receivedTemplate" },
        ];

        for (const {key, listId, templateId} of sections) {
            const usersData: UserData[] = await fetchUserInformation(friendIds[key]);
            const list = document.getElementById(listId);
            const template = document.getElementById(templateId) as HTMLTemplateElement | null;

            if (usersData.length && list && template) {
                list.innerHTML = '';
                for (const userData of  usersData) {
                    const clone = template.content.cloneNode(true) as HTMLElement;
                    const img = clone.querySelector("img");
                    if (img && userData.avatar) img.src = userData.avatar;
                    const name = clone.querySelector(".name");
                    if (name) name.textContent = userData.nickName;
                    if (userData.online)
                        clone.querySelector(".online")?.classList.remove('hidden');
                    if (key === "receivedIds") {
                        clone.querySelector(".accept-btn")?.addEventListener("click", () => addFriend(userData.nickName));
                        clone.querySelector(".decline-btn")?.addEventListener("click", () => removeFriend(userData.nickName));
                    }
                    list.appendChild(clone);
                }
            }
        }
    } catch (err) {
        console.error("Erreur dans friendList():", err);
    }
}
