import {handleConnection} from "./front.js"
import {loadPart} from "./insert.js"
import {sseConnection} from "./serverSentEvent.js"

export function register(button: HTMLElement): void {
    button.addEventListener("click", async () => {
        const myForm = document.getElementById("myForm") as HTMLFormElement
        const formData = new FormData(myForm)
        const data = Object.fromEntries(formData as unknown as Iterable<readonly any[]>)
        const response = await fetch(`/user-management/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        const result = await response.json()
        if (result.token) {
            localStorage.setItem('token', result.token)
            localStorage.setItem('nickName', result.nickName)
            sseConnection()
            loadPart("/connected")
            handleConnection(true)
        } else {
            console.log("ErrorDisplay : nickname, email or password are not respecting my authority")
        }
    })
}

export function login(button: HTMLElement): void {
    button.addEventListener("click", async () => {
        const myForm = document.getElementById("myForm") as HTMLFormElement
        const formData = new FormData(myForm)
        const data = Object.fromEntries(formData as unknown as Iterable<readonly any[]>)
        const response = await fetch(`/user-management/login`, {
           method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })

        if (response.ok) {
            const data = await response.json()
            localStorage.setItem('nickName', data.nickName)
            if (!data.token || data.token.empty) {
                return await loadPart("/2fa")
            }
            localStorage.setItem('token', data.token)
            await loadPart("/connected")
            handleConnection(true)
            getAvatar();
            await sseConnection()
        } else {
            const errorData = await response.json()
            const loginError = document.getElementById("LoginError") as HTMLSpanElement
            loginError.textContent = errorData?.error ?? "An error occurred"
            loginError.classList.remove("hidden")
        }
    })
}

export interface Person {
    id: number,
    // email: string
    imageUrl: string
}

type FriendList = {
    acceptedIds: number[]
    pendingIds: number[]
    receivedIds: number[]
}

export async function friendList() {
    try {
        const friendlist = await getFriendList() as FriendList
        const noFriend = document.getElementById("noFriend") as HTMLHeadingElement
        // if (!friendlist || !friendlist.receivedNickName.length) {
        //     if (noFriend)
        //         noFriend.classList.remove("hidden")
        // } else {
        //     const receivedPeople: Person[] = friendlist.receivedNickName.map(nickname => ({
        //         name: nickname,
        //         imageUrl: '../images/login.png'
        //     }))
        //     const receivedList = document.getElementById("receivedList")
        //     const receivedTemplate = document.getElementById("receivedTemplate") as HTMLTemplateElement
        //
        //     if (receivedList && receivedTemplate) {
        //         receivedList.innerHTML = ''
        //         receivedPeople.forEach(person => {
        //             const clone = receivedTemplate.content.cloneNode(true) as HTMLElement
        //             const img = clone.querySelector("img")!
        //             const name = clone.querySelector(".name")!
        //
        //             img.src = person.imageUrl
        //             img.alt = person.name
        //             name.textContent = person.name
        //
        //             receivedList.appendChild(clone)
        //         })
        //     }
        // }
        if (!friendlist || !friendlist.pendingIds.length) {
            if (noFriend)
                noFriend.classList.remove("hidden")
        } else {
            const requestPeople: Person[] = friendlist.pendingIds.map(id => ({
                id: id,
                imageUrl: '../images/login.png'
            }))
            const requestList = document.getElementById("requestList")
            const requestTemplate = document.getElementById("requestTemplate") as HTMLTemplateElement

            if (requestList && requestTemplate) {
                requestList.innerHTML = ''
                requestPeople.forEach(person => {
                    const clone = requestTemplate.content.cloneNode(true) as HTMLElement
                    const img = clone.querySelector("img")!
                    const name = clone.querySelector(".name")!

                    img.src = person.imageUrl
                    name.textContent = person.id.toString()

                    requestList.appendChild(clone)
                })
            }
        }
        // if (!friendlist || !friendlist.acceptedNickName.length) {
        //     if (noFriend)
        //         noFriend.classList.remove("hidden")
        // } else {
        //     const acceptedPeople: Person[] = friendlist.acceptedNickName.map(nickname => ({
        //         name: nickname,
        //         imageUrl: '../images/login.png'
        //     }))
        //     const acceptedList = document.getElementById("acceptedList")
        //     const acceptedTemplate = document.getElementById("acceptedTemplate") as HTMLTemplateElement
        //
        //     if (acceptedList && acceptedTemplate) {
        //         acceptedList.innerHTML = ''
        //         acceptedPeople.forEach(person => {
        //             const clone = acceptedTemplate.content.cloneNode(true) as HTMLElement
        //             const img = clone.querySelector("img")!
        //             const name = clone.querySelector(".name")!
        //
        //             img.src = person.imageUrl
        //             img.alt = person.name
        //             name.textContent = person.name
        //
        //             acceptedList.appendChild(clone)
        //         })
        //     }
        // }
    } catch (err) {
        console.error("Erreur dans friendList():", err)
    }
}

export async function profile(nickName: HTMLElement, email: HTMLElement) {
    try {
        const token = localStorage.getItem('token')
        if (!token) {
            console.error('token missing')
            return
        }
        const response = await fetch(`/user-management/getProfile`, {
        method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token
            }
        })
        const data = await response.json()
        if (response.ok) {
            nickName.innerText = data.nickName
            email.innerText = data.email
        }
    }
    catch (err) {
        console.error(err)
    }
}

export async function init2fa() {
    try {
        const token = localStorage.getItem('token')
        if (!token) {
            console.error('token missing')
            return
        }
        const response = await fetch(`/user-management/2faInit`, {
           method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token
            }
        })
        if (response.ok)
            return await response.text()
        return await response.json()
    } catch (err) {
        console.error(err)
    }
}

export async function verify2fa(secret: string) {
    try {
        const nickName = localStorage.getItem('nickName')
        const response = await fetch(`/user-management/2faVerify`, {
        method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({secret: secret, nickName: nickName})
        })
        if (response.ok) {
            const result = await response.json()
            localStorage.setItem('token', result.token)
            loadPart('/home')
            handleConnection(true)
            localStorage.setItem('factor', 'true') //Add hidden factor 2fa
        }
        else {
            const errorData = await response.json()
            console.error('errorDisplay', errorData)
        }
    } catch (err) {
        console.error(err)
    }
}


export async function addFriend(friendNickName: string) {
    try {
        const token = localStorage.getItem('token')
        const response = await fetch(`/social/add`, {
           method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
            body: JSON.stringify({friendNickName: friendNickName})
        })
        //Clear input after use
        const input = document.getElementById('friendNameIpt') as HTMLInputElement
        input.value = ''
        input.focus()
        if (!response.ok) {
            const error = await response.json()
            console.error(error.error)
            return false
        }
        return true
    } catch (error) {
        console.error(error)
        return false
    }
}

export async function removeFriend(friendNickName: string): Promise<boolean> {
    try {
        const token = localStorage.getItem('token')
        const response = await fetch(`/social/remove`, {
        method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
            body: JSON.stringify({friendNickName: friendNickName}),
        })

        if (!response.ok) {
            const error = await response.json()
            console.error(error.error)
            return false
        }
        return true
    } catch (error) {
        console.error(error)
        return false
    }
}

export async function getFriendList(): Promise<FriendList | undefined> {
    try {
        const token = localStorage.getItem('token')
        const response = await fetch(`/social/List`, {
           method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
        })
        if (!response.ok) {
            const error = await response.json()
            console.error(error.error)
            return undefined
        }
        return await response.json()
    } catch (error) {
        console.error(error)
    }
}

// @ts-ignore
const toBase64 = (file: any) => new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = reject
    reader.readAsDataURL(file)
})

export async function setAvatar(target: HTMLInputElement) {
    try {
        if (target.files && target.files[0]) {
            const file: File = target.files[0]
            const base64File: string = await toBase64(file) as string

            const token = localStorage.getItem('token')
            const response = await fetch(`/user-management/updatePicture`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'authorization': 'Bearer ' + token,
                },
                body: JSON.stringify({pictureURL: base64File})
            })
            if (!response.ok) {
                const error = await response.json()
                console.error(error.error)
            }
            else {
                localStorage.setItem('avatar', base64File)
                updateAvatar('avatarProfile', base64File);
                updateAvatar('avatar', base64File);
            }
        }
    } catch (err) {
        console.error(err)
    }
}

// @ts-ignore
export async function getAvatar() {
    try {
        const avatarImg = document.getElementById('avatar') as HTMLImageElement
        const avatar = localStorage.getItem('avatar')
        if (avatar) {
            avatarImg.src = avatar
            return ;
        }

        const token = localStorage.getItem('token')
        const response = await fetch(`/user-management/getPicture`, {
           method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
        })
        if (!response.ok) {
            const error = await response.json()
            console.error(error.error)
            return ;
        }
        const img = await response.json()
        if (img.url) {
            avatarImg.src = img.url
            localStorage.setItem('avatar', avatarImg.src)
        }
        else
            avatarImg.src = '../images/login.png'

    } catch (error) {
        console.error(error)
    }
}

export const updateAvatar = (id: string, src: string) => {
    const img = document.getElementById(id) as HTMLImageElement | null;
    if (img) {
        img.src = src;
    } else {
        console.warn(`Element with id '${id}' not found.`);
    }
};

export async function joinTournament() {
    try {
        const token = localStorage.getItem('token')
        const response = await fetch(`/tournament/join`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
        })

        if (!response.ok) {
            const error = await response.json()
            console.error(error.error)
        }
    } catch (error) {
        console.error(error)
    }
}

export async function getTournamentList(): Promise<string[] | undefined> {
    try {
        const token = localStorage.getItem('token')
        const response = await fetch(`/tournament/getList`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
        })
        if (!response.ok) {
            const error = await response.json()
            console.error(error.error)
            return undefined
        }
        return await response.json()
    } catch (error) {
        console.error(error)
    }
}

export async function launchTournament() {
    try {
        const token = localStorage.getItem('token')
        const response = await fetch(`/tournament/launch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'authorization': 'Bearer ' + token,
            },
        })

        if (!response.ok) {
            const error = await response.json()
            console.error(error.error)
        }
    } catch (error) {
        console.error(error)
    }
}
