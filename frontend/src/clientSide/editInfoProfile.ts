import {setNickName} from "./user.js";

export function editProfile() {
    document.getElementById("editProfileButton")?.addEventListener("click", async () => {
        const nickInput = document.getElementById("profileNickName") as HTMLInputElement | null;
        if (nickInput) {
            const newNickName = nickInput.value.trim();
            await setNickName(newNickName);
            sessionStorage.setItem("nickName", newNickName);
        }
    });
}