import { setNickName } from './user.js'

export function editProfile() {
	// @ts-ignore
	document.getElementById('editProfileButton')?.addEventListener('click', async () => {
		const nickInput = document.getElementById('profileNickName') as HTMLInputElement | null
		const emailInput = document.getElementById('profileEmail') as HTMLInputElement | null
		if (nickInput && emailInput) {
			const newNickName = nickInput.value.trim()
			const newEmail = emailInput.value.trim()
			console.log('New nickname:', newNickName)
			console.log('New email:', newEmail)
			await setNickName(newNickName)
			sessionStorage.setItem('nickName', newNickName)
			// setPassword();
			// setEmail();
			// sessionStorage.setItem("email", newEmail);
		}
	})
}