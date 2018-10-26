function setup() {
    const posts = Array.from(document.getElementsByClassName('post-link'));
    posts.forEach(post => {
        post.addEventListener('click', (e) => {
            let location = window.location;
            let postUrl = post.getAttribute('href');
            let newLocation = "";
            if (post.hasAttribute('foreign') && post.getAttribute('foreign')) {
                newLocation= postUrl;
            } else {
                newLocation = `http://${location['host']}${postUrl}`;
            }
            window.location = newLocation;
        });
    });

    const nav = document.getElementById('nav-site-pages');
    const anchors = nav.getElementsByTagName('a');
    const [_, current] = window.location.pathname.split('/');
    const domain = window.location.origin;
    for (const anchor of anchors) {
        const relative = anchor.href.substring(domain.length + 1);
        if (relative === current) {
            anchor.classList.add('active');
        } else {
            anchor.classList.remove('active');
        }
    }

    if (document.querySelector('.list-of-contents') !== null) {
        createContentsList();
    }
}

function createContentsList() {
    const sections = enumerateSections();
    const contents = document.querySelector('.list-of-contents ul');

    for (const sectionDef of sections) {
        const anchorLink = document.createElement('a');
        anchorLink.href = `#${sectionDef['id']}`;
        anchorLink.appendChild(document.createTextNode(sectionDef['header']));
        const listItem = document.createElement('li');
        listItem.classList += " post-section-link list-unstyled";
        listItem.appendChild(anchorLink);
        contents.appendChild(listItem);
    }
}

/*
 * Appends numerical indicies to each section's header. Returns total number of
 * discovered sections.
 */
function enumerateSections(selector = "h4.header") {
    let content = [];
    const postSectionHeaders = Array.from(document.querySelectorAll(selector));

    for (const [index, header] of postSectionHeaders.entries()) {
        const headerText = `[${index}] ${header.textContent}`;
        const contentEntry = {
            id: header.id,
            header: headerText,
        }
        content.push(contentEntry);
        header.innerHTML = '';
        header.appendChild(document.createTextNode(headerText));
    }

    return content;
}
