def studentMatching(studentPreferences, advisorPreferences, quota):
    studentAssignment = {}  # applicant id to advisor
    advisorAssignment = {}  # advisor id to ***list*** of students
    """IMPLEMENT METHOD HERE"""
    # Be careful because students are proposing

    # Advisors propose and students accept
    open_advisors = set(advisorPreferences.keys())

    # proceed until all students are matched
    while open_advisors:

        proposals = {}

        # student proposes to advisor
        for student in studentPreferences:

            if student not in studentAssignment and studentPreferences[student]:
                advisor = studentPreferences[student].pop(0)
                proposals[advisor] = proposals.get(advisor, []) + [student]

        # no proposals left
        if not proposals:
            break

        # advisor accepts proposal
        for advisor, applicants in proposals.items():
            current_accepted_students = advisorAssignment.get(advisor, [])
            # Tentatively add every acceptable students
            current_accepted_students.extend(
                [s for s in applicants if s in advisorPreferences[advisor]]
            )
            # Sort based on the preference
            current_accepted_students.sort(
                key=lambda x: advisorPreferences[advisor].index(x)
            )
            # Keep the top quota students
            current_accepted_students = current_accepted_students[: quota[advisor]]
            rejected_students = current_accepted_students[quota[advisor] :]

            # Update the assignment
            advisorAssignment[advisor] = current_accepted_students
            for student in current_accepted_students:
                studentAssignment[student] = advisor
            for student in rejected_students:
                if student in studentAssignment:
                    studentAssignment.pop(student)

    return [studentAssignment, advisorAssignment]

