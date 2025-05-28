import {setNickName} from "./user.js";

function strictNickname(input: string) {
    return input.replace(/[^\w]/g, ''); // garde uniquement [a-zA-Z0-9_]
}

export function editProfile() {
    document.getElementById("editProfileButton")?.addEventListener("click", async () => {
        const nickInput = document.getElementById("profileNickName") as HTMLInputElement | null;
        if (nickInput) {
            const newNickName = strictNickname(nickInput.value.trim());
            await setNickName(newNickName);
        }
    });
}