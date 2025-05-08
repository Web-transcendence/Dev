import {setNickName} from "./user.js";

export function editProfile() {
    document.getElementById("editProfileButton")?.addEventListener("click", () => {
        const nickInput = document.getElementById("profileNickName") as HTMLInputElement | null;
        const emailInput = document.getElementById("profileEmail") as HTMLInputElement | null;
        if (nickInput && emailInput) {
            const newNickName = nickInput.value.trim();
            const newEmail = emailInput.value.trim();
            console.log("New nickname:", newNickName);
            console.log("New email:", newEmail);
            setNickName(newNickName);
            localStorage.setItem("nickName", newNickName);
            // setPassword();
            // setEmail();
            // localStorage.setItem("email", newEmail);
        }
    });
}