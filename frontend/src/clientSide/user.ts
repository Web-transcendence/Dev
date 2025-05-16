import {navigate} from "./front.js"
import {loadPart} from "./insert.js"
import {sseConnection} from "./serverSentEvent.js"
import {DispayNotification} from "./notificationHandler.js"

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
        console.log( result)
        if (result.nickName) {
            console.log(`yyyyyyyyyyyyy`)
            sessionStorage.setItem('id', result.id)
            sessionStorage.setItem('nickName', result.nickName)
            await navigate('/connected')
            await getAvatar();
            await sseConnection()
        } else {
            DispayNotification(result.error , { type: "error" });
        }
    })
}

export function login(button: HTMLElement): void {
    button.addEventListener("click", async () => {
        const myForm = document.getElementById("myForm") as HTMLFormElement
        const formData = new FormData(myForm)
        const data = Object.fromEntries(formData as unknown as Iterable<readonly any[]>)
        console.log(`sssssssssss`)
        const response = await fetch(`/user-management/login`, {
           method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })

        if (response.ok) {
            const data = await response.json()
            sessionStorage.setItem('id', data.id)
            sessionStorage.setItem('nickName', data.nickName)
            if (data.connected === false) {
                return await loadPart("/factor")
            }
            navigate('/connected')
            await getAvatar();
            await sseConnection()
        } else {
            const errorData = await response.json()
            DispayNotification('Wrong password or nickname', { type: "error" });
        }
    })
}


export async function profile() {
    try {
        const response = await fetch(`/user-management/privateProfile`, {
        method: 'GET',
            headers: {
                'Content-Type': 'application/json'         }
        })
        const data = await response.json()
        if (response.ok) {
            sessionStorage.setItem('id', data.id)
            sessionStorage.setItem('nickName', data.nickName)
            sessionStorage.setItem('email', data.email)
            if (data.avatar)
                sessionStorage.setItem('avatar', data.avatar)
            const nameInput = document.getElementById("profileNickName");
            if (nameInput instanceof HTMLInputElement) nameInput.value = data.nickName;
            const emailInput = document.getElementById("profileEmail");
            if (emailInput instanceof HTMLInputElement) emailInput.value = data.email;
        }
    }
    catch (err) {
        console.error(err)
    }
}


export type UserData = {
    id: number,
    nickName: string,
    avatar: string,
    online: boolean
}

export const fetchUserInformation = async (ids: number[]): Promise<UserData[]> => {
    if (!ids)
        throw new Error('ids missing')
    const response = await fetch(`/user-management/userInformation`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'},
        body: JSON.stringify({ids})
    })
    if (response.ok) {
        const {usersData} = await response.json() as {usersData: UserData[]}
        return usersData
    }
    else {
        const error = await response.json()
        console.log(error)
        //notify error
        throw error
    }
}

export async function init2fa() {
    try {
        const response = await fetch(`/user-management/2faInit`, {
           method: 'GET',
            headers: {
                'Content-Type': 'application/json'            }
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
        const nickName = sessionStorage.getItem('nickName')
        const response = await fetch(`/user-management/2faVerify`, {
        method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({secret: secret, nickName: nickName})
        })
        if (response.ok) {
            sessionStorage.setItem('activeFA', 'true')
            DispayNotification('You have enabled two-factor authentication.');
            await navigate('/home')
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
        const response = await fetch(`/social/add`, {
           method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({friendNickName: friendNickName})
        })
        //Clear input after use
        const input = document.getElementById('friendNameIpt') as HTMLInputElement
        input.value = ''
        input.focus()
        if (!response.ok) {
            const error = await response.json()
            console.error('ERROR addFriend', error.error)
            DispayNotification(error.error, { type: "error" })
            return false
        }
        DispayNotification('Successfully Invited')
        return true
    } catch (error) {
        console.error(error)
        return false
    }
}

export async function removeFriend(friendNickName: string): Promise<boolean> {
    try {
        const response = await fetch(`/social/remove`, {
        method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({friendNickName: friendNickName}),
        })

        if (!response.ok) {
            const error = await response.json()
            console.error(error.error)
            return false
        }
        DispayNotification('Successfully Removed')
        return true
    } catch (error) {
        console.error(error)
        return false
    }
}

export type FriendIds = {
    acceptedIds: number[];
    pendingIds: number[];
    receivedIds: number[];
}

export async function getFriendList() : Promise<FriendIds> {
    const response = await fetch(`/social/list`, {
       method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    })
    if (!response.ok) {
        const error = await response.json()
        console.error(error.error)
        throw error
    }
    return await response.json()
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
        console.log('files', target.files)
        if (target.files && target.files[0]) {
            const file: File = target.files[0]
            const base64File: string = await toBase64(file) as string

            const response = await fetch(`/user-management/updatePicture`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({pictureURL: base64File})
            })
            if (!response.ok) {
                const error = await response.json()
                console.error(error.error)
                DispayNotification(error.error, { type: "error" })
            }
            else {
                sessionStorage.setItem('avatar', base64File)
                updateAvatar('avatarProfile', base64File);
                updateAvatar('avatar', base64File);
                DispayNotification('New Avatar !')
            }
        }
    } catch (err) {
        DispayNotification('Not a File', { type: "error" })
    }
}

// @ts-ignore
export async function getAvatar() {
    try {
        const avatarImg = document.getElementById('avatar') as HTMLImageElement
        const avatar = sessionStorage.getItem('avatar')
        if (avatar) {
            avatarImg.src = avatar
            return ;
        }

        const response = await fetch(`/user-management/getPicture`, {
           method: 'GET',
            headers: {
                'Content-Type': 'application/json',
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
            console.log('avatar', avatarImg.src)
            sessionStorage.setItem('avatar', avatarImg.src)
        }
        else
            avatarImg.src = '../images/login.png'

    } catch (error) {
        console.error(error)
    }
}

export async function setPassword(newPassword: string) {
    try {
        const response = await fetch(`/user-management/setPassword`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({password: newPassword}),
        })
        if (!response.ok) {
            console.error('failed')
        }
        else
            console.log('success')

    } catch (error) {
        console.error(error)
    }
}

export async function setNickName(newNickName: string) {
    try {
        const response = await fetch(`/user-management/setNickName`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({nickName: newNickName}),
        })
        if (!response.ok) {
            console.error('failed')
            DispayNotification('Bad input', { type: "error" } )
        } else {
            console.log('success')
            DispayNotification('New Nickname')
        }
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

export async function getTournamentList() {
    try {
        const response = await fetch(`/tournament/getList`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        })
        if (!response.ok) {
            const error = await response.json()
            console.error(error.error)
            DispayNotification("Register To join Tournaments", { type: "error" })
            return undefined
        }
        return await response.json()
    } catch (error) {
        console.error(error)
    }
}
