import {addFriend, fetchUserInformation, FriendIds, getFriendList, removeFriend, UserData} from "./user.js";
import { CreateFriendLi } from "./serverSentEvent.js";

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
                    const item = clone.querySelector("li");
                    if (item) item.id = `friendId-${userData.id}`;
                    const img = clone.querySelector("img");
                    if (img && userData.avatar) img.src = userData.avatar;
                    const name = clone.querySelector(".name");
                    if (name) name.textContent = userData.nickName;
                    if (userData.online && key === "acceptedIds")
                        clone.querySelector(".online")?.classList.remove('hidden');
                    if (key === "receivedIds") {
                        clone.querySelector(".accept-btn")?.addEventListener("click", async () => {
                            if (await addFriend(userData.nickName))
                                CreateFriendLi(userData.id, "acceptedList", "acceptedTemplate")
                        });
                        clone.querySelector(".decline-btn")?.addEventListener("click", async () => {
                            if (await removeFriend(userData.nickName))
                                document.getElementById(`friendId-${userData.id}`)?.remove();
                        });
                    }
                    if (key === "acceptedIds") {
                        clone.querySelector(".inviteBtn")?.addEventListener("click", async () => {
                            const modal = document.getElementById("myModal") as HTMLDivElement | undefined;
                            if (!modal) return ;
                            modal.classList.remove("hidden");
                            modal.classList.add("flex");
                            modal.addEventListener("click", (event) => {
                                if (event.target === modal) {
                                    modal.classList.remove("flex");
                                    modal.classList.add("hidden");
                                }
                            });
                            document.getElementById("closeModalBtn")?.addEventListener("click", () => {
                                modal.classList.remove("flex");
                                modal.classList.add("hidden");
                            });
                        });
                    }
                    list.appendChild(clone);
                }
            }
        }
    } catch (err) {
        console.error("Erreur dans friendList():", err);
    }
}
