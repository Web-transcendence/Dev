import {addFriend, getFriendList, removeFriend} from "./user.js";

export interface Person {
    name: string
    // email: string
    imageUrl: string
}

type FriendList = {
    acceptedNickName: string[]
    pendingNickName: string[]
    receivedNickName: string[]
}

export async function friendList() {
    try {
        const friendlist = await getFriendList() as FriendList;
        const noFriend = document.getElementById("noFriend") as HTMLHeadingElement;

        const sections = [
            { key: "receivedNickName", listId: "receivedList", templateId: "receivedTemplate" },
            { key: "pendingNickName", listId: "requestList", templateId: "requestTemplate" },
            { key: "acceptedNickName", listId: "acceptedList", templateId: "acceptedTemplate" },
        ];

        let hasFriends = false;

        sections.forEach(({ key, listId, templateId }) => {
            const nicknames = (friendlist as any)[key] as string[];
            const list = document.getElementById(listId);
            const template = document.getElementById(templateId) as HTMLTemplateElement;

            if (nicknames?.length && list && template) {
                hasFriends = true;
                list.innerHTML = '';
                nicknames.forEach(name => {
                    const clone = template.content.cloneNode(true) as HTMLElement;
                    const img = clone.querySelector("img")!;
                    const text = clone.querySelector(".name")!;
                    img.src = '../images/login.png';
                    img.alt = name;
                    text.textContent = name;
                    if (key === "receivedNickName") {
                        clone.querySelector(".accept-btn")?.addEventListener("click", () => addFriend(name));
                        // clone.querySelector(".decline-btn")?.addEventListener("click", () => removeFriend(name));
                    }
                    list.appendChild(clone);
                });
            }
        });

        if (!hasFriends && noFriend) {
            noFriend.classList.remove("hidden");
        }
    } catch (err) {
        console.error("Erreur dans friendList():", err);
    }
}
